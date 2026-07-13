import json
import re
from collections.abc import Generator
from datetime import UTC, datetime
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.cluster_role_binding import ClusterRoleBinding
from ocp_resources.config_map import ConfigMap
from ocp_resources.namespace import Namespace
from ocp_resources.notebook import Notebook
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.resource import Resource
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from pytest_testconfig import config as py_config
from semver import Version
from timeout_sampler import TimeoutExpiredError

from tests.workbenches.notebooks_server.controller.utils import (
    WORKBENCH_TRUSTED_CA_BUNDLE_NAME,
    MutatingWebhookConfiguration,
    OAuthClient,
    StatefulSet,
    build_notebook_dict,
    resolve_notebook_image,
)
from utilities import constants
from utilities.general import collect_pod_information
from utilities.infra import create_ns, get_product_version
from utilities.resources.http_route import HTTPRoute
from utilities.resources.reference_grant import ReferenceGrant

LOGGER = structlog.get_logger(name=__name__)

UPGRADE_NAMESPACE = "upgrade-workbenches"
UPGRADE_NOTEBOOK_NAME = "upgrade-workbenches"
UPGRADE_STOPPED_NOTEBOOK_NAME = "upgrade-wb-stopped"
NEW_NOTEBOOK_NAME = "upgrade-wb-new"
NOTEBOOK_MUTATING_WEBHOOK_NAME = "odh-notebook-controller-mutating-webhook-configuration"
UPGRADE_BASELINE_CM_NAME = "upgrade-workbenches-baseline"
ODH_TRUSTED_CA_BUNDLE_NAME = "odh-trusted-ca-bundle"

OAUTH_PROXY_CONTAINER = "oauth-proxy"
OAUTH_FINALIZER = "notebook-oauth-client-finalizer.opendatahub.io"
OAUTH_VOLUMES = frozenset({"oauth-config", "oauth-client", "tls-certificates"})
TORNADO_SETTINGS_PATTERN = re.compile(pattern=r"\s*--ServerApp\.tornado_settings=[^\n]*")


def migrate_notebook_to_3x(notebook: Notebook, client: DynamicClient) -> None:
    """Patch a 2.x Notebook CR for the 3.x auth model.

    Replicates the rhoai-upgrade-helpers `patch` command:
    1. Add inject-auth annotation, remove inject-oauth and oauth-logout-url
    2. Remove oauth-proxy container from spec
    3. Remove oauth-related volumes (oauth-config, oauth-client, tls-certificates)
    4. Remove notebook-oauth-client-finalizer finalizer
    5. Strip --ServerApp.tornado_settings from NOTEBOOK_ARGS env var
    6. Delete the StatefulSet to force recreation by the controller
    """
    spec = notebook.instance.spec.template.spec
    containers = spec.containers or []
    volumes = spec.volumes or []
    finalizers = notebook.instance.metadata.finalizers or []

    patched_containers = [c for c in containers if c.name != OAUTH_PROXY_CONTAINER]
    patched_volumes = [v for v in volumes if v.name not in OAUTH_VOLUMES]
    patched_finalizers = [f for f in finalizers if f != OAUTH_FINALIZER]

    for container in patched_containers:
        if not container.env:
            continue
        for env_var in container.env:
            if env_var.name == "NOTEBOOK_ARGS" and env_var.value:
                env_var.value = TORNADO_SETTINGS_PATTERN.sub(repl="", string=env_var.value)

    annotations = dict(notebook.instance.metadata.annotations or {})
    annotations.pop("notebooks.opendatahub.io/inject-oauth", None)
    annotations.pop("notebooks.opendatahub.io/oauth-logout-url", None)
    annotations["notebooks.opendatahub.io/inject-auth"] = "true"

    notebook.update({
        "metadata": {
            "name": notebook.name,
            "annotations": annotations,
            "finalizers": patched_finalizers,
        },
        "spec": {
            "template": {
                "spec": {
                    "containers": [c.to_dict() for c in patched_containers],
                    "volumes": [v.to_dict() for v in patched_volumes],
                }
            }
        },
    })
    LOGGER.info(f"Patched Notebook CR '{notebook.name}' for 3.x auth model")

    sts = StatefulSet(
        client=client,
        name=notebook.name,
        namespace=notebook.namespace,
    )
    if sts.exists:
        sts.delete(wait=True)
        LOGGER.info(f"Deleted StatefulSet '{notebook.name}' to force recreation")


def cleanup_legacy_oauth_resources(
    notebook_name: str,
    namespace: str,
    client: DynamicClient,
    admin_client: DynamicClient,
) -> None:
    """Remove leftover OAuth resources after migration.

    Replicates the rhoai-upgrade-helpers `cleanup` command:
    - Route: {name}
    - Service: {name}-tls
    - Secret: {name}-oauth-client, {name}-oauth-config, {name}-tls
    - OAuthClient: {name}-{namespace}-oauth-client (cluster-scoped)
    """
    resources_to_delete: list[Resource] = [
        Route(client=client, name=notebook_name, namespace=namespace),
        Service(client=client, name=f"{notebook_name}-tls", namespace=namespace),
        Secret(client=client, name=f"{notebook_name}-oauth-client", namespace=namespace),
        Secret(client=client, name=f"{notebook_name}-oauth-config", namespace=namespace),
        Secret(client=client, name=f"{notebook_name}-tls", namespace=namespace),
    ]

    for resource in resources_to_delete:
        if resource.exists:
            resource.delete(wait=True)
            LOGGER.info(f"Deleted {resource.kind} '{resource.name}' in '{namespace}'")

    oauth_client = OAuthClient(
        client=admin_client,
        name=f"{notebook_name}-{namespace}-oauth-client",
    )
    if oauth_client.exists:
        oauth_client.delete(wait=True)
        LOGGER.info(f"Deleted OAuthClient '{oauth_client.name}'")


@pytest.fixture(scope="session")
def upgrade_notebook_namespace(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[Namespace, Any, Any]:
    """Namespace for workbench upgrade tests."""
    ns = Namespace(client=unprivileged_client, name=UPGRADE_NAMESPACE)

    if pytestconfig.option.post_upgrade:
        yield ns
        if teardown_resources:
            ns.client = admin_client
            ns.clean_up()
    else:
        with create_ns(
            admin_client=admin_client,
            unprivileged_client=unprivileged_client,
            name=UPGRADE_NAMESPACE,
            add_dashboard_label=True,
            teardown=teardown_resources,
        ) as ns:
            yield ns


@pytest.fixture(scope="session")
def upgrade_notebook_pvc(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    upgrade_notebook_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    """PVC for the upgrade workbench notebook."""
    pvc_kwargs = {
        "client": unprivileged_client,
        "name": UPGRADE_NOTEBOOK_NAME,
        "namespace": upgrade_notebook_namespace.name,
    }

    if pytestconfig.option.post_upgrade:
        yield PersistentVolumeClaim(**pvc_kwargs)
    else:
        with PersistentVolumeClaim(
            **pvc_kwargs,
            label={constants.Labels.OpenDataHub.DASHBOARD: "true"},
            accessmodes=PersistentVolumeClaim.AccessMode.RWO,
            size="1Gi",
            volume_mode=PersistentVolumeClaim.VolumeMode.FILE,
            teardown=teardown_resources,
        ) as pvc:
            yield pvc


@pytest.fixture(scope="session")
def upgrade_notebook_image(admin_client: DynamicClient) -> str:
    """Resolves the notebook image path for upgrade tests."""
    return resolve_notebook_image(admin_client=admin_client)


@pytest.fixture(scope="session")
def upgrade_notebook(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
    upgrade_notebook_namespace: Namespace,
    upgrade_notebook_pvc: PersistentVolumeClaim,
    upgrade_notebook_image: str,
    teardown_resources: bool,
) -> Generator[Notebook, Any, Any]:
    """Notebook CR for upgrade tests."""
    notebook_kwargs = {
        "client": unprivileged_client,
        "name": UPGRADE_NOTEBOOK_NAME,
        "namespace": upgrade_notebook_namespace.name,
    }

    if pytestconfig.option.post_upgrade:
        nb = Notebook(**notebook_kwargs)
        yield nb
        if teardown_resources:
            nb.client = admin_client
            nb.clean_up()
    else:
        notebook_dict = build_notebook_dict(
            namespace=upgrade_notebook_namespace.name,
            name=UPGRADE_NOTEBOOK_NAME,
            image_path=upgrade_notebook_image,
        )

        with Notebook(client=unprivileged_client, kind_dict=notebook_dict, teardown=teardown_resources) as nb:
            yield nb


@pytest.fixture(scope="session")
def upgrade_notebook_pod(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    upgrade_notebook: Notebook,
) -> Pod:
    """Notebook pod for upgrade tests.

    Pre-upgrade: waits for the pod to reach Ready state.
    Post-upgrade: wraps the existing pod (expected to still be running).
    """
    notebook_pod = Pod(
        client=unprivileged_client,
        namespace=upgrade_notebook.namespace,
        name=f"{upgrade_notebook.name}-0",
    )

    if pytestconfig.option.post_upgrade:
        return notebook_pod

    try:
        notebook_pod.wait()
        notebook_pod.wait_for_condition(
            condition=Pod.Condition.READY,
            status=Pod.Condition.Status.TRUE,
            timeout=300,
        )
    except (TimeoutError, TimeoutExpiredError) as e:
        if notebook_pod.exists:
            collect_pod_information(notebook_pod)
            raise AssertionError(
                f"Pod '{upgrade_notebook.name}-0' failed to reach Ready state "
                f"within (300) seconds.\n"
                f"Original Error: {e}\n"
                f"Pod information collected to must-gather directory for debugging."
            ) from e

        raise AssertionError(f"Pod '{upgrade_notebook.name}-0' was not created. Check notebook controller logs.") from e

    return notebook_pod


@pytest.fixture(scope="session")
def upgrade_notebook_statefulset(
    unprivileged_client: DynamicClient,
    upgrade_notebook: Notebook,
) -> StatefulSet:
    """StatefulSet owned by the Notebook CR."""
    return StatefulSet(
        client=unprivileged_client,
        name=upgrade_notebook.name,
        namespace=upgrade_notebook.namespace,
    )


@pytest.fixture(scope="session")
def upgrade_notebook_service(
    unprivileged_client: DynamicClient,
    upgrade_notebook: Notebook,
) -> Service:
    """Service owned by the Notebook CR."""
    return Service(
        client=unprivileged_client,
        name=upgrade_notebook.name,
        namespace=upgrade_notebook.namespace,
    )


@pytest.fixture(scope="session")
def upgrade_notebook_httproute(
    admin_client: DynamicClient,
    upgrade_notebook: Notebook,
) -> HTTPRoute:
    """HTTPRoute for the notebook in the applications (controller) namespace."""
    httproute_name = f"nb-{upgrade_notebook.namespace}-{upgrade_notebook.name}"
    return HTTPRoute(
        client=admin_client,
        name=httproute_name,
        namespace=py_config["applications_namespace"],
    )


@pytest.fixture(scope="session")
def upgrade_notebook_reference_grant(
    admin_client: DynamicClient,
    upgrade_notebook_namespace: Namespace,
) -> ReferenceGrant:
    """ReferenceGrant in the notebook namespace allowing cross-namespace HTTPRoute access."""
    return ReferenceGrant(
        client=admin_client,
        name="notebook-httproute-access",
        namespace=upgrade_notebook_namespace.name,
    )


@pytest.fixture(scope="session")
def auth_proxy_service(
    unprivileged_client: DynamicClient,
    upgrade_notebook: Notebook,
) -> Service:
    """kube-rbac-proxy Service for the notebook."""
    return Service(
        client=unprivileged_client,
        name=f"{upgrade_notebook.name}-kube-rbac-proxy",
        namespace=upgrade_notebook.namespace,
    )


@pytest.fixture(scope="session")
def auth_proxy_configmap(
    unprivileged_client: DynamicClient,
    upgrade_notebook: Notebook,
) -> ConfigMap:
    """kube-rbac-proxy ConfigMap for the notebook."""
    return ConfigMap(
        client=unprivileged_client,
        name=f"{upgrade_notebook.name}-kube-rbac-proxy-config",
        namespace=upgrade_notebook.namespace,
    )


@pytest.fixture(scope="session")
def auth_delegator_crb(
    admin_client: DynamicClient,
    upgrade_notebook: Notebook,
) -> ClusterRoleBinding:
    """auth-delegator ClusterRoleBinding for the notebook's kube-rbac-proxy."""
    return ClusterRoleBinding(
        client=admin_client,
        name=f"{upgrade_notebook.name}-rbac-{upgrade_notebook.namespace}-auth-delegator",
    )


@pytest.fixture(scope="session")
def stopped_auth_proxy_service(
    unprivileged_client: DynamicClient,
    stopped_notebook: Notebook,
) -> Service:
    """kube-rbac-proxy Service for the stopped notebook."""
    return Service(
        client=unprivileged_client,
        name=f"{stopped_notebook.name}-kube-rbac-proxy",
        namespace=stopped_notebook.namespace,
    )


@pytest.fixture(scope="session")
def stopped_auth_proxy_configmap(
    unprivileged_client: DynamicClient,
    stopped_notebook: Notebook,
) -> ConfigMap:
    """kube-rbac-proxy ConfigMap for the stopped notebook."""
    return ConfigMap(
        client=unprivileged_client,
        name=f"{stopped_notebook.name}-kube-rbac-proxy-config",
        namespace=stopped_notebook.namespace,
    )


@pytest.fixture(scope="session")
def stopped_auth_delegator_crb(
    admin_client: DynamicClient,
    stopped_notebook: Notebook,
) -> ClusterRoleBinding:
    """auth-delegator ClusterRoleBinding for the stopped notebook's kube-rbac-proxy."""
    return ClusterRoleBinding(
        client=admin_client,
        name=f"{stopped_notebook.name}-rbac-{stopped_notebook.namespace}-auth-delegator",
    )


@pytest.fixture(scope="session")
def stopped_notebook_pvc(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    upgrade_notebook_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    """PVC for the stopped notebook upgrade scenario."""
    pvc_kwargs = {
        "client": unprivileged_client,
        "name": UPGRADE_STOPPED_NOTEBOOK_NAME,
        "namespace": upgrade_notebook_namespace.name,
    }

    if pytestconfig.option.post_upgrade:
        yield PersistentVolumeClaim(**pvc_kwargs)
    else:
        with PersistentVolumeClaim(
            **pvc_kwargs,
            label={constants.Labels.OpenDataHub.DASHBOARD: "true"},
            accessmodes=PersistentVolumeClaim.AccessMode.RWO,
            size="1Gi",
            volume_mode=PersistentVolumeClaim.VolumeMode.FILE,
            teardown=teardown_resources,
        ) as pvc:
            yield pvc


@pytest.fixture(scope="session")
def stopped_notebook(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
    upgrade_notebook_namespace: Namespace,
    stopped_notebook_pvc: PersistentVolumeClaim,
    upgrade_notebook_image: str,
    teardown_resources: bool,
) -> Generator[Notebook, Any, Any]:
    """Notebook CR that is stopped before upgrade via kubeflow-resource-stopped annotation."""
    notebook_kwargs = {
        "client": unprivileged_client,
        "name": UPGRADE_STOPPED_NOTEBOOK_NAME,
        "namespace": upgrade_notebook_namespace.name,
    }

    if pytestconfig.option.post_upgrade:
        nb = Notebook(**notebook_kwargs)
        yield nb
        if teardown_resources:
            nb.client = admin_client
            nb.clean_up()
    else:
        notebook_dict = build_notebook_dict(
            namespace=upgrade_notebook_namespace.name,
            name=UPGRADE_STOPPED_NOTEBOOK_NAME,
            image_path=upgrade_notebook_image,
        )

        with Notebook(client=unprivileged_client, kind_dict=notebook_dict, teardown=teardown_resources) as nb:
            yield nb


@pytest.fixture(scope="session")
def stopped_notebook_statefulset(
    unprivileged_client: DynamicClient,
    stopped_notebook: Notebook,
) -> StatefulSet:
    """StatefulSet for the stopped notebook."""
    return StatefulSet(
        client=unprivileged_client,
        name=stopped_notebook.name,
        namespace=stopped_notebook.namespace,
    )


@pytest.fixture(scope="session")
def stopped_notebook_pre_upgrade_shutdown(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    stopped_notebook: Notebook,
    stopped_notebook_statefulset: StatefulSet,
) -> None:
    """Pre-upgrade stopped notebook state: annotation applied, pod terminated, replicas=0.

    No-op during post-upgrade runs.
    """
    if pytestconfig.option.post_upgrade:
        return

    notebook_pod = Pod(
        client=unprivileged_client,
        namespace=stopped_notebook.namespace,
        name=f"{stopped_notebook.name}-0",
    )

    try:
        notebook_pod.wait()
        notebook_pod.wait_for_condition(
            condition=Pod.Condition.READY,
            status=Pod.Condition.Status.TRUE,
            timeout=300,
        )
    except (TimeoutError, TimeoutExpiredError) as e:
        if notebook_pod.exists:
            collect_pod_information(notebook_pod)
            raise AssertionError(
                f"Pod '{stopped_notebook.name}-0' failed to reach Ready state "
                f"before stop. Cannot proceed with upgrade test. Original error: {e}"
            ) from e

        raise AssertionError(f"Pod '{stopped_notebook.name}-0' was not created. Check notebook controller logs.") from e

    stop_timestamp = datetime.now(tz=UTC).strftime(format="%Y-%m-%dT%H:%M:%SZ")
    stopped_notebook.update({
        "metadata": {
            "name": stopped_notebook.name,
            "annotations": {"kubeflow-resource-stopped": stop_timestamp},
        }
    })
    LOGGER.info(
        f"Stopped notebook '{stopped_notebook.name}' via kubeflow-resource-stopped annotation "
        f"with timestamp '{stop_timestamp}'"
    )

    notebook_pod.wait_deleted(timeout=120)
    LOGGER.info(f"Pod '{notebook_pod.name}' terminated after stop annotation")

    replicas = stopped_notebook_statefulset.instance.spec.replicas
    assert replicas == 0, (
        f"StatefulSet '{stopped_notebook_statefulset.name}' has {replicas} replicas after stop, expected 0"
    )
    LOGGER.info(f"StatefulSet '{stopped_notebook_statefulset.name}' confirmed at 0 replicas")


@pytest.fixture(scope="session")
def workbench_trusted_ca_bundle(
    unprivileged_client: DynamicClient,
    upgrade_notebook_namespace: Namespace,
) -> ConfigMap:
    """The workbench-trusted-ca-bundle ConfigMap created by the ODH controller."""
    return ConfigMap(
        client=unprivileged_client,
        name=WORKBENCH_TRUSTED_CA_BUNDLE_NAME,
        namespace=upgrade_notebook_namespace.name,
    )


@pytest.fixture(scope="session")
def odh_trusted_ca_bundle(
    admin_client: DynamicClient,
) -> ConfigMap:
    """The odh-trusted-ca-bundle ConfigMap in the applications namespace (source of trust)."""
    return ConfigMap(
        client=admin_client,
        name=ODH_TRUSTED_CA_BUNDLE_NAME,
        namespace=py_config["applications_namespace"],
    )


@pytest.fixture(scope="session")
def notebook_mutating_webhook(
    admin_client: DynamicClient,
) -> MutatingWebhookConfiguration:
    """The MutatingWebhookConfiguration for the ODH notebook controller."""
    return MutatingWebhookConfiguration(
        client=admin_client,
        name=NOTEBOOK_MUTATING_WEBHOOK_NAME,
    )


@pytest.fixture(scope="session")
def capture_notebook_baseline(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    upgrade_notebook: Notebook,
    upgrade_notebook_pod: Pod,
    upgrade_notebook_statefulset: StatefulSet,
    upgrade_notebook_service: Service,
    upgrade_notebook_httproute: HTTPRoute,
    stopped_notebook: Notebook,
    stopped_notebook_pre_upgrade_shutdown: None,
    workbench_trusted_ca_bundle: ConfigMap,
    odh_trusted_ca_bundle: ConfigMap,
) -> None:
    """Capture notebook resource metadata to a ConfigMap before upgrade.

    No-op during post-upgrade runs.
    """
    if pytestconfig.option.post_upgrade:
        return

    creation_timestamp = upgrade_notebook_pod.instance.metadata.creationTimestamp
    assert creation_timestamp, f"Pod '{upgrade_notebook_pod.name}' has no creationTimestamp in metadata"

    notebook_generation = upgrade_notebook.instance.metadata.generation
    sts_generation = upgrade_notebook_statefulset.instance.metadata.generation
    service_spec = upgrade_notebook_service.instance.spec
    service_ports = json.dumps(service_spec.ports, sort_keys=True, default=str)
    service_selector = json.dumps(service_spec.selector, sort_keys=True, default=str)
    upgrade_notebook_httproute.wait()
    assert upgrade_notebook_httproute.exists, (
        f"HTTPRoute '{upgrade_notebook_httproute.name}' not found in "
        f"'{upgrade_notebook_httproute.namespace}' during baseline capture"
    )
    httproute_generation = upgrade_notebook_httproute.instance.metadata.generation

    stopped_annotation = stopped_notebook.instance.metadata.annotations.get("kubeflow-resource-stopped")

    assert workbench_trusted_ca_bundle.exists, (
        f"ConfigMap '{WORKBENCH_TRUSTED_CA_BUNDLE_NAME}' not found in "
        f"'{upgrade_notebook.namespace}' during baseline capture"
    )
    ca_bundle_resource_version = workbench_trusted_ca_bundle.instance.metadata.resourceVersion

    assert odh_trusted_ca_bundle.exists, (
        f"ConfigMap '{ODH_TRUSTED_CA_BUNDLE_NAME}' not found in "
        f"'{py_config['applications_namespace']}' during baseline capture"
    )
    odh_ca_bundle_resource_version = odh_trusted_ca_bundle.instance.metadata.resourceVersion

    source_version = get_product_version(admin_client=admin_client)

    containers = upgrade_notebook_pod.instance.spec.containers
    sidecar_names = {"oauth-proxy", "kube-rbac-proxy"}
    main_container = next((container for container in containers if container.name not in sidecar_names), None)
    notebook_image = main_container.image if main_container else ""

    baseline = {
        "ntb_creation_timestamp": creation_timestamp,
        "notebook_generation": notebook_generation,
        "statefulset_generation": sts_generation,
        "service_ports": service_ports,
        "service_selector": service_selector,
        "httproute_generation": httproute_generation,
        "stopped_annotation_value": stopped_annotation,
        "ca_bundle_resource_version": ca_bundle_resource_version,
        "odh_ca_bundle_resource_version": odh_ca_bundle_resource_version,
        "source_rhoai_version": str(source_version),
        "notebook_image": notebook_image,
    }

    ConfigMap(
        client=admin_client,
        name=UPGRADE_BASELINE_CM_NAME,
        namespace=UPGRADE_NAMESPACE,
        data={"baseline": json.dumps(baseline)},
    ).deploy()

    LOGGER.info(f"Saved notebook upgrade baseline: {baseline}")


@pytest.fixture(scope="session")
def upgrade_notebook_baseline(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
) -> dict[str, Any]:
    """Load the pre-upgrade notebook baseline from the ConfigMap.

    Returns an empty dict during pre-upgrade runs.
    """
    if not pytestconfig.option.post_upgrade:
        return {}

    cm = ConfigMap(
        client=admin_client,
        name=UPGRADE_BASELINE_CM_NAME,
        namespace=UPGRADE_NAMESPACE,
    )

    assert cm.exists, (
        f"Baseline ConfigMap '{UPGRADE_BASELINE_CM_NAME}' not found in namespace '{UPGRADE_NAMESPACE}'. "
        f"Ensure pre-upgrade tests ran successfully."
    )

    cm_data = cm.instance.data or {}
    raw = cm_data.get("baseline")
    assert raw, f"Baseline ConfigMap '{UPGRADE_BASELINE_CM_NAME}' has no 'baseline' key in data."

    return json.loads(raw)


@pytest.fixture(scope="session")
def new_notebook_pvc(
    unprivileged_client: DynamicClient,
    upgrade_notebook_namespace: Namespace,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    """PVC for the post-upgrade new notebook creation test."""
    with PersistentVolumeClaim(
        client=unprivileged_client,
        name=NEW_NOTEBOOK_NAME,
        namespace=upgrade_notebook_namespace.name,
        label={constants.Labels.OpenDataHub.DASHBOARD: "true"},
        accessmodes=PersistentVolumeClaim.AccessMode.RWO,
        size="1Gi",
        volume_mode=PersistentVolumeClaim.VolumeMode.FILE,
        teardown=True,
    ) as pvc:
        yield pvc


@pytest.fixture(scope="session")
def new_notebook(
    unprivileged_client: DynamicClient,
    upgrade_notebook_namespace: Namespace,
    upgrade_notebook_image: str,
    new_notebook_pvc: PersistentVolumeClaim,
) -> Generator[Notebook, Any, Any]:
    """Fresh Notebook CR created post-upgrade to verify controller functionality."""
    notebook_dict = build_notebook_dict(
        namespace=upgrade_notebook_namespace.name,
        name=NEW_NOTEBOOK_NAME,
        image_path=upgrade_notebook_image,
    )

    with Notebook(client=unprivileged_client, kind_dict=notebook_dict, teardown=True) as nb:
        yield nb


@pytest.fixture(scope="session")
def new_notebook_pod(
    unprivileged_client: DynamicClient,
    new_notebook: Notebook,
) -> Pod:
    """Pod for the post-upgrade new notebook; waits for Ready state."""
    notebook_pod = Pod(
        client=unprivileged_client,
        namespace=new_notebook.namespace,
        name=f"{new_notebook.name}-0",
    )

    try:
        notebook_pod.wait()
        notebook_pod.wait_for_condition(
            condition=Pod.Condition.READY,
            status=Pod.Condition.Status.TRUE,
            timeout=300,
        )
    except (TimeoutError, TimeoutExpiredError) as e:
        if notebook_pod.exists:
            collect_pod_information(notebook_pod)
            raise AssertionError(
                f"New notebook pod '{new_notebook.name}-0' failed to reach Ready state "
                f"within (300) seconds on upgraded platform.\n"
                f"Original error: {e}"
            ) from e

        raise AssertionError(
            f"New notebook pod '{new_notebook.name}-0' was not created on upgraded platform.\nOriginal error: {e}"
        ) from e

    return notebook_pod


@pytest.fixture(scope="session")
def new_notebook_statefulset(
    unprivileged_client: DynamicClient,
    new_notebook: Notebook,
) -> StatefulSet:
    """StatefulSet owned by the post-upgrade new Notebook CR."""
    return StatefulSet(
        client=unprivileged_client,
        name=new_notebook.name,
        namespace=new_notebook.namespace,
    )


@pytest.fixture(scope="session")
def new_notebook_service(
    unprivileged_client: DynamicClient,
    new_notebook: Notebook,
) -> Service:
    """Service owned by the post-upgrade new Notebook CR."""
    return Service(
        client=unprivileged_client,
        name=new_notebook.name,
        namespace=new_notebook.namespace,
    )


@pytest.fixture(scope="session")
def new_notebook_httproute(
    admin_client: DynamicClient,
    new_notebook: Notebook,
    upgrade_notebook_namespace: Namespace,
) -> HTTPRoute:
    """HTTPRoute created for the post-upgrade new notebook."""
    httproute_name = f"nb-{upgrade_notebook_namespace.name}-{new_notebook.name}"
    return HTTPRoute(
        client=admin_client,
        name=httproute_name,
        namespace=py_config["applications_namespace"],
    )


@pytest.fixture(scope="session")
def new_notebook_auth_proxy_service(
    unprivileged_client: DynamicClient,
    new_notebook: Notebook,
) -> Service:
    """kube-rbac-proxy Service for the post-upgrade new notebook."""
    return Service(
        client=unprivileged_client,
        name=f"{new_notebook.name}-kube-rbac-proxy",
        namespace=new_notebook.namespace,
    )


@pytest.fixture(scope="session")
def new_notebook_auth_proxy_configmap(
    unprivileged_client: DynamicClient,
    new_notebook: Notebook,
) -> ConfigMap:
    """kube-rbac-proxy ConfigMap for the post-upgrade new notebook."""
    return ConfigMap(
        client=unprivileged_client,
        name=f"{new_notebook.name}-kube-rbac-proxy-config",
        namespace=new_notebook.namespace,
    )


@pytest.fixture(scope="session")
def new_notebook_auth_delegator_crb(
    admin_client: DynamicClient,
    new_notebook: Notebook,
) -> ClusterRoleBinding:
    """auth-delegator ClusterRoleBinding for the post-upgrade new notebook."""
    return ClusterRoleBinding(
        client=admin_client,
        name=f"{new_notebook.name}-rbac-{new_notebook.namespace}-auth-delegator",
    )


@pytest.fixture(scope="session")
def source_rhoai_version(
    upgrade_notebook_baseline: dict[str, Any],
) -> Version | None:
    """The RHOAI version that was installed before the upgrade.

    Returns None during pre-upgrade runs or if the baseline does not contain version info.
    """
    raw = upgrade_notebook_baseline.get("source_rhoai_version")
    if not raw:
        return None
    return Version.parse(version=raw)


@pytest.fixture(scope="session")
def is_migration_from_2x(
    source_rhoai_version: Version | None,
) -> bool:
    """Whether the upgrade is from RHOAI 2.x to 3.x (a major migration boundary)."""
    if source_rhoai_version is None:
        return False
    return source_rhoai_version.major < 3


@pytest.fixture(scope="session")
def self_migrate(pytestconfig: pytest.Config) -> bool:
    """Whether tests should perform 2.x-to-3.x migration themselves."""
    return pytestconfig.option.self_migrate


@pytest.fixture(scope="session")
def upgrade_notebook_route(
    unprivileged_client: DynamicClient,
    upgrade_notebook: Notebook,
) -> Route:
    """OpenShift Route for the running notebook (2.x-style routing)."""
    return Route(
        client=unprivileged_client,
        name=upgrade_notebook.name,
        namespace=upgrade_notebook.namespace,
    )


@pytest.fixture(scope="session")
def stopped_notebook_route(
    unprivileged_client: DynamicClient,
    stopped_notebook: Notebook,
) -> Route:
    """OpenShift Route for the stopped notebook (2.x-style routing)."""
    return Route(
        client=unprivileged_client,
        name=stopped_notebook.name,
        namespace=stopped_notebook.namespace,
    )


@pytest.fixture(scope="session")
def restart_stopped_notebook(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    admin_client: DynamicClient,
    stopped_notebook: Notebook,
    self_migrate: bool,
) -> Pod:
    """Migrate and start the stopped notebook, returning the Ready pod.

    In self-migrate mode: patches Notebook CR (removes oauth-proxy, changes annotation),
    deletes StatefulSet, cleans up legacy resources, then starts.
    In external mode: assumes migration scripts already ran; just removes stop annotation.
    """
    assert pytestconfig.option.post_upgrade, "restart_stopped_notebook fixture is only valid during post-upgrade"

    if self_migrate:
        migrate_notebook_to_3x(notebook=stopped_notebook, client=unprivileged_client)
        cleanup_legacy_oauth_resources(
            notebook_name=stopped_notebook.name,
            namespace=stopped_notebook.namespace,
            client=unprivileged_client,
            admin_client=admin_client,
        )

    stopped_notebook.update({
        "metadata": {
            "name": stopped_notebook.name,
            "annotations": {"kubeflow-resource-stopped": None},
        }
    })
    LOGGER.info(f"Removed kubeflow-resource-stopped from '{stopped_notebook.name}' to start")

    notebook_pod = Pod(
        client=unprivileged_client,
        namespace=stopped_notebook.namespace,
        name=f"{stopped_notebook.name}-0",
    )

    try:
        notebook_pod.wait()
        notebook_pod.wait_for_condition(
            condition=Pod.Condition.READY,
            status=Pod.Condition.Status.TRUE,
            timeout=Timeout.TIMEOUT_5MIN,
        )
    except (TimeoutError, TimeoutExpiredError) as e:
        if notebook_pod.exists:
            collect_pod_information(notebook_pod)
            raise AssertionError(
                f"Stopped notebook pod '{stopped_notebook.name}-0' failed to reach Ready state "
                f"after restart within {Timeout.TIMEOUT_5MIN} seconds.\nOriginal error: {e}"
            ) from e

        raise AssertionError(
            f"Stopped notebook pod '{stopped_notebook.name}-0' was not created after restart.\nOriginal error: {e}"
        ) from e

    return notebook_pod


@pytest.fixture(scope="session")
def restart_running_notebook(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    admin_client: DynamicClient,
    upgrade_notebook: Notebook,
    self_migrate: bool,
) -> Pod:
    """Migrate and restart the running notebook via stop-migrate-start cycle.

    In self-migrate mode: stops the notebook, patches CR, cleans up legacy resources, then starts.
    In external mode: assumes notebook was already stopped + patched by pipeline; just starts it.
    """
    assert pytestconfig.option.post_upgrade, "restart_running_notebook fixture is only valid during post-upgrade"

    if self_migrate:
        stop_timestamp = datetime.now(tz=UTC).strftime(format="%Y-%m-%dT%H:%M:%SZ")
        upgrade_notebook.update({
            "metadata": {
                "name": upgrade_notebook.name,
                "annotations": {"kubeflow-resource-stopped": stop_timestamp},
            }
        })
        LOGGER.info(f"Stopped running notebook '{upgrade_notebook.name}' with timestamp '{stop_timestamp}'")

        old_pod = Pod(
            client=unprivileged_client,
            namespace=upgrade_notebook.namespace,
            name=f"{upgrade_notebook.name}-0",
        )
        old_pod.wait_deleted(timeout=Timeout.TIMEOUT_2MIN)
        LOGGER.info(f"Pod '{old_pod.name}' terminated after stop annotation")

        migrate_notebook_to_3x(notebook=upgrade_notebook, client=unprivileged_client)
        cleanup_legacy_oauth_resources(
            notebook_name=upgrade_notebook.name,
            namespace=upgrade_notebook.namespace,
            client=unprivileged_client,
            admin_client=admin_client,
        )

    upgrade_notebook.update({
        "metadata": {
            "name": upgrade_notebook.name,
            "annotations": {"kubeflow-resource-stopped": None},
        }
    })
    LOGGER.info(f"Removed kubeflow-resource-stopped from '{upgrade_notebook.name}' to start")

    new_pod = Pod(
        client=unprivileged_client,
        namespace=upgrade_notebook.namespace,
        name=f"{upgrade_notebook.name}-0",
    )

    try:
        new_pod.wait()
        new_pod.wait_for_condition(
            condition=Pod.Condition.READY,
            status=Pod.Condition.Status.TRUE,
            timeout=Timeout.TIMEOUT_5MIN,
        )
    except (TimeoutError, TimeoutExpiredError) as e:
        if new_pod.exists:
            collect_pod_information(new_pod)
            raise AssertionError(
                f"Running notebook pod '{upgrade_notebook.name}-0' failed to reach Ready state "
                f"after restart within {Timeout.TIMEOUT_5MIN} seconds.\nOriginal error: {e}"
            ) from e

        raise AssertionError(
            f"Running notebook pod '{upgrade_notebook.name}-0' was not created after restart.\nOriginal error: {e}"
        ) from e

    return new_pod
