import json
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
from ocp_resources.service import Service
from pytest_testconfig import config as py_config

from tests.workbenches.notebooks_server.controller.upgrade.kueue_constants import (
    NEW_KUEUE_NOTEBOOK_NAME,
    UPGRADE_KUEUE_BASELINE_CM_NAME,
    UPGRADE_KUEUE_CLUSTER_QUEUE_NAME,
    UPGRADE_KUEUE_HARDWARE_PROFILE_NAME,
    UPGRADE_KUEUE_LOCAL_QUEUE_NAME,
    UPGRADE_KUEUE_NAMESPACE,
    UPGRADE_KUEUE_NOTEBOOK_NAME,
    UPGRADE_KUEUE_RESOURCE_FLAVOR_NAME,
    UPGRADE_KUEUE_STOPPED_NOTEBOOK_NAME,
)
from tests.workbenches.notebooks_server.controller.utils import (
    KUBEFLOW_STOPPED_ANNOTATION,
    WORKBENCH_TRUSTED_CA_BUNDLE_NAME,
    HardwareProfile,
    MutatingWebhookConfiguration,
    StatefulSet,
    build_notebook_dict,
    resolve_notebook_image,
    wait_for_notebook_pod_ready,
)
from utilities import constants
from utilities.infra import create_ns
from utilities.kueue_utils import (
    KUEUE_CLUSTER_QUEUE_LABEL,
    KUEUE_LOCAL_QUEUE_LABEL,
    KUEUE_MANAGED_LABEL,
    KUEUE_QUEUE_NAME_LABEL,
    ClusterQueue,
    LocalQueue,
    ResourceFlavor,
    create_cluster_queue,
    create_local_queue,
    create_resource_flavor,
)
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

    wait_for_notebook_pod_ready(notebook_pod=notebook_pod, context="Upgrade notebook")

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

    wait_for_notebook_pod_ready(notebook_pod=notebook_pod, context="Stopped notebook (pre-stop)")

    stop_timestamp = datetime.now(tz=UTC).strftime(format="%Y-%m-%dT%H:%M:%SZ")
    stopped_notebook.update({
        "metadata": {
            "name": stopped_notebook.name,
            "annotations": {KUBEFLOW_STOPPED_ANNOTATION: stop_timestamp},
        }
    })
    LOGGER.info(
        f"Stopped notebook '{stopped_notebook.name}' via {KUBEFLOW_STOPPED_ANNOTATION} annotation "
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

    stopped_annotation = stopped_notebook.instance.metadata.annotations.get(KUBEFLOW_STOPPED_ANNOTATION)

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

    wait_for_notebook_pod_ready(notebook_pod=notebook_pod, context="New notebook (post-upgrade)")

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


# ---------------------------------------------------------------------------
# Kueue Upgrade Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def upgrade_kueue_namespace(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[Namespace, Any, Any]:
    """Namespace with kueue.openshift.io/managed=true for kueue upgrade tests."""
    ns = Namespace(client=unprivileged_client, name=UPGRADE_KUEUE_NAMESPACE)

    if pytestconfig.option.post_upgrade:
        yield ns
        if teardown_resources:
            ns.client = admin_client
            ns.clean_up()
    else:
        with create_ns(
            admin_client=admin_client,
            unprivileged_client=unprivileged_client,
            name=UPGRADE_KUEUE_NAMESPACE,
            add_kueue_label=True,
            teardown=teardown_resources,
        ) as ns:
            yield ns


@pytest.fixture(scope="session")
def upgrade_kueue_resource_flavor(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[ResourceFlavor, Any, Any]:
    """ResourceFlavor for kueue upgrade tests."""
    if pytestconfig.option.post_upgrade:
        rf = ResourceFlavor(
            client=admin_client,
            name=UPGRADE_KUEUE_RESOURCE_FLAVOR_NAME,
        )
        yield rf
        if teardown_resources:
            rf.clean_up()
    else:
        with create_resource_flavor(
            client=admin_client,
            name=UPGRADE_KUEUE_RESOURCE_FLAVOR_NAME,
            teardown=teardown_resources,
        ) as resource_flavor:
            yield resource_flavor


@pytest.fixture(scope="session")
def upgrade_kueue_cluster_queue(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    upgrade_kueue_resource_flavor: ResourceFlavor,
    teardown_resources: bool,
) -> Generator[ClusterQueue, Any, Any]:
    """ClusterQueue for kueue upgrade tests with generous quotas."""
    if pytestconfig.option.post_upgrade:
        cq = ClusterQueue(
            client=admin_client,
            name=UPGRADE_KUEUE_CLUSTER_QUEUE_NAME,
        )
        yield cq
        if teardown_resources:
            cq.clean_up()
    else:
        resource_groups = [
            {
                "coveredResources": ["cpu", "memory"],
                "flavors": [
                    {
                        "name": upgrade_kueue_resource_flavor.name,
                        "resources": [
                            {"name": "cpu", "nominalQuota": "4"},
                            {"name": "memory", "nominalQuota": "8Gi"},
                        ],
                    }
                ],
            }
        ]

        with create_cluster_queue(
            client=admin_client,
            name=UPGRADE_KUEUE_CLUSTER_QUEUE_NAME,
            resource_groups=resource_groups,
            namespace_selector={"matchLabels": {"kubernetes.io/metadata.name": UPGRADE_KUEUE_NAMESPACE}},
            teardown=teardown_resources,
        ) as cluster_queue:
            yield cluster_queue


@pytest.fixture(scope="session")
def upgrade_kueue_local_queue(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    upgrade_kueue_namespace: Namespace,
    upgrade_kueue_cluster_queue: ClusterQueue,
    teardown_resources: bool,
) -> Generator[LocalQueue, Any, Any]:
    """LocalQueue for kueue upgrade tests."""
    if pytestconfig.option.post_upgrade:
        lq = LocalQueue(
            client=admin_client,
            name=UPGRADE_KUEUE_LOCAL_QUEUE_NAME,
            namespace=upgrade_kueue_namespace.name,
            cluster_queue=UPGRADE_KUEUE_CLUSTER_QUEUE_NAME,
        )
        yield lq
        if teardown_resources:
            lq.clean_up()
    else:
        with create_local_queue(
            client=admin_client,
            name=UPGRADE_KUEUE_LOCAL_QUEUE_NAME,
            cluster_queue=upgrade_kueue_cluster_queue.name,
            namespace=upgrade_kueue_namespace.name,
            teardown=teardown_resources,
        ) as local_queue:
            yield local_queue


@pytest.fixture(scope="session")
def upgrade_kueue_hardware_profile(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    upgrade_kueue_namespace: Namespace,
    upgrade_kueue_local_queue: LocalQueue,
    teardown_resources: bool,
) -> Generator[HardwareProfile, Any, Any]:
    """HardwareProfile with Kueue queue-based scheduling for upgrade tests."""
    hwp_kwargs = {
        "client": admin_client,
        "name": UPGRADE_KUEUE_HARDWARE_PROFILE_NAME,
        "namespace": upgrade_kueue_namespace.name,
    }

    if pytestconfig.option.post_upgrade:
        hwp = HardwareProfile(**hwp_kwargs)
        yield hwp
        if teardown_resources:
            hwp.clean_up()
    else:
        kind_dict: dict[str, Any] = {
            "apiVersion": "infrastructure.opendatahub.io/v1",
            "kind": "HardwareProfile",
            "metadata": {
                "name": UPGRADE_KUEUE_HARDWARE_PROFILE_NAME,
                "namespace": upgrade_kueue_namespace.name,
            },
            "spec": {
                "identifiers": [
                    {
                        "displayName": "CPU",
                        "identifier": "cpu",
                        "minCount": "500m",
                        "maxCount": "2",
                        "defaultCount": "1",
                        "resourceType": "CPU",
                    },
                    {
                        "displayName": "Memory",
                        "identifier": "memory",
                        "minCount": "512Mi",
                        "maxCount": "4Gi",
                        "defaultCount": "1Gi",
                        "resourceType": "Memory",
                    },
                ],
                "scheduling": {
                    "type": "Queue",
                    "kueue": {
                        "localQueueName": upgrade_kueue_local_queue.name,
                    },
                },
            },
        }

        with HardwareProfile(client=admin_client, kind_dict=kind_dict, teardown=teardown_resources) as hwp:
            yield hwp


@pytest.fixture(scope="session")
def upgrade_kueue_notebook_pvc(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    upgrade_kueue_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    """PVC for the kueue upgrade notebook."""
    pvc_kwargs = {
        "client": unprivileged_client,
        "name": UPGRADE_KUEUE_NOTEBOOK_NAME,
        "namespace": upgrade_kueue_namespace.name,
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
def upgrade_kueue_notebook(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
    upgrade_kueue_namespace: Namespace,
    upgrade_kueue_notebook_pvc: PersistentVolumeClaim,
    upgrade_kueue_hardware_profile: HardwareProfile,
    upgrade_notebook_image: str,
    teardown_resources: bool,
) -> Generator[Notebook, Any, Any]:
    """Notebook CR referencing a Kueue-enabled HardwareProfile for upgrade tests."""
    notebook_kwargs = {
        "client": unprivileged_client,
        "name": UPGRADE_KUEUE_NOTEBOOK_NAME,
        "namespace": upgrade_kueue_namespace.name,
    }

    if pytestconfig.option.post_upgrade:
        nb = Notebook(**notebook_kwargs)
        yield nb
        if teardown_resources:
            nb.client = admin_client
            nb.clean_up()
    else:
        notebook_dict = build_notebook_dict(
            namespace=upgrade_kueue_namespace.name,
            name=UPGRADE_KUEUE_NOTEBOOK_NAME,
            image_path=upgrade_notebook_image,
            extra_annotations={
                "opendatahub.io/hardware-profile-name": upgrade_kueue_hardware_profile.name,
            },
            resources={},
        )

        with Notebook(client=unprivileged_client, kind_dict=notebook_dict, teardown=teardown_resources) as nb:
            yield nb


@pytest.fixture(scope="session")
def upgrade_kueue_notebook_pod(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    upgrade_kueue_notebook: Notebook,
) -> Pod:
    """Kueue-managed notebook pod for upgrade tests.

    Pre-upgrade: waits for Ready state.
    Post-upgrade: wraps the existing pod.
    """
    notebook_pod = Pod(
        client=unprivileged_client,
        namespace=upgrade_kueue_notebook.namespace,
        name=f"{upgrade_kueue_notebook.name}-0",
    )

    if pytestconfig.option.post_upgrade:
        return notebook_pod

    wait_for_notebook_pod_ready(notebook_pod=notebook_pod, context="Kueue notebook")

    return notebook_pod


@pytest.fixture(scope="session")
def upgrade_kueue_notebook_statefulset(
    unprivileged_client: DynamicClient,
    upgrade_kueue_notebook: Notebook,
) -> StatefulSet:
    """StatefulSet owned by the kueue Notebook CR."""
    return StatefulSet(
        client=unprivileged_client,
        name=upgrade_kueue_notebook.name,
        namespace=upgrade_kueue_notebook.namespace,
    )


@pytest.fixture(scope="session")
def upgrade_kueue_stopped_notebook_pvc(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    upgrade_kueue_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    """PVC for the stopped kueue notebook."""
    pvc_kwargs = {
        "client": unprivileged_client,
        "name": UPGRADE_KUEUE_STOPPED_NOTEBOOK_NAME,
        "namespace": upgrade_kueue_namespace.name,
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
def upgrade_kueue_stopped_notebook(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
    upgrade_kueue_namespace: Namespace,
    upgrade_kueue_stopped_notebook_pvc: PersistentVolumeClaim,
    upgrade_kueue_hardware_profile: HardwareProfile,
    upgrade_notebook_image: str,
    teardown_resources: bool,
) -> Generator[Notebook, Any, Any]:
    """Notebook CR referencing a Kueue-enabled HardwareProfile, stopped before upgrade."""
    notebook_kwargs = {
        "client": unprivileged_client,
        "name": UPGRADE_KUEUE_STOPPED_NOTEBOOK_NAME,
        "namespace": upgrade_kueue_namespace.name,
    }

    if pytestconfig.option.post_upgrade:
        nb = Notebook(**notebook_kwargs)
        yield nb
        if teardown_resources:
            nb.client = admin_client
            nb.clean_up()
    else:
        notebook_dict = build_notebook_dict(
            namespace=upgrade_kueue_namespace.name,
            name=UPGRADE_KUEUE_STOPPED_NOTEBOOK_NAME,
            image_path=upgrade_notebook_image,
            extra_annotations={
                "opendatahub.io/hardware-profile-name": upgrade_kueue_hardware_profile.name,
            },
            resources={},
        )

        with Notebook(client=unprivileged_client, kind_dict=notebook_dict, teardown=teardown_resources) as nb:
            yield nb


@pytest.fixture(scope="session")
def upgrade_kueue_stopped_notebook_statefulset(
    unprivileged_client: DynamicClient,
    upgrade_kueue_stopped_notebook: Notebook,
) -> StatefulSet:
    """StatefulSet for the stopped kueue notebook."""
    return StatefulSet(
        client=unprivileged_client,
        name=upgrade_kueue_stopped_notebook.name,
        namespace=upgrade_kueue_stopped_notebook.namespace,
    )


@pytest.fixture(scope="session")
def upgrade_kueue_stopped_pre_upgrade_shutdown(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    upgrade_kueue_stopped_notebook: Notebook,
    upgrade_kueue_stopped_notebook_statefulset: StatefulSet,
) -> None:
    """Pre-upgrade: stop the kueue notebook, verify pod terminated and replicas=0.

    No-op during post-upgrade runs.
    """
    if pytestconfig.option.post_upgrade:
        return

    notebook_pod = Pod(
        client=unprivileged_client,
        namespace=upgrade_kueue_stopped_notebook.namespace,
        name=f"{upgrade_kueue_stopped_notebook.name}-0",
    )

    wait_for_notebook_pod_ready(notebook_pod=notebook_pod, context="Kueue stopped notebook (pre-stop)")

    stop_timestamp = datetime.now(tz=UTC).strftime(format="%Y-%m-%dT%H:%M:%SZ")
    upgrade_kueue_stopped_notebook.update({
        "metadata": {
            "name": upgrade_kueue_stopped_notebook.name,
            "annotations": {KUBEFLOW_STOPPED_ANNOTATION: stop_timestamp},
        }
    })
    LOGGER.info(f"Stopped kueue notebook '{upgrade_kueue_stopped_notebook.name}' with timestamp '{stop_timestamp}'")

    notebook_pod.wait_deleted(timeout=120)
    LOGGER.info(f"Pod '{notebook_pod.name}' terminated after stop annotation")

    replicas = upgrade_kueue_stopped_notebook_statefulset.instance.spec.replicas
    assert replicas == 0, (
        f"StatefulSet '{upgrade_kueue_stopped_notebook_statefulset.name}' "
        f"has {replicas} replicas after stop, expected 0"
    )


@pytest.fixture(scope="session")
def capture_kueue_baseline(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    upgrade_kueue_notebook: Notebook,
    upgrade_kueue_notebook_pod: Pod,
    upgrade_kueue_resource_flavor: ResourceFlavor,
    upgrade_kueue_cluster_queue: ClusterQueue,
    upgrade_kueue_local_queue: LocalQueue,
    upgrade_kueue_stopped_notebook: Notebook,
    upgrade_kueue_stopped_pre_upgrade_shutdown: None,
) -> None:
    """Capture kueue notebook resource metadata to a ConfigMap before upgrade.

    No-op during post-upgrade runs.
    """
    if pytestconfig.option.post_upgrade:
        return

    creation_timestamp = upgrade_kueue_notebook_pod.instance.metadata.creationTimestamp
    assert creation_timestamp, f"Kueue notebook pod '{upgrade_kueue_notebook_pod.name}' has no creationTimestamp"

    pod_labels = upgrade_kueue_notebook_pod.instance.metadata.labels or {}
    notebook_generation = upgrade_kueue_notebook.instance.metadata.generation

    for _label in (KUEUE_MANAGED_LABEL, KUEUE_QUEUE_NAME_LABEL, KUEUE_CLUSTER_QUEUE_LABEL, KUEUE_LOCAL_QUEUE_LABEL):
        assert pod_labels.get(_label), (
            f"Pre-upgrade kueue pod '{upgrade_kueue_notebook_pod.name}' missing label '{_label}'; "
            f"refusing to capture an empty baseline. Labels: {list(pod_labels.keys())}"
        )

    stopped_annotation = upgrade_kueue_stopped_notebook.instance.metadata.annotations.get(KUBEFLOW_STOPPED_ANNOTATION)

    baseline = {
        "pod_creation_timestamp": creation_timestamp,
        "pod_kueue_managed_label": pod_labels.get(KUEUE_MANAGED_LABEL, ""),
        "pod_queue_name_label": pod_labels.get(KUEUE_QUEUE_NAME_LABEL, ""),
        "pod_cluster_queue_label": pod_labels.get(KUEUE_CLUSTER_QUEUE_LABEL, ""),
        "pod_local_queue_label": pod_labels.get(KUEUE_LOCAL_QUEUE_LABEL, ""),
        "notebook_generation": notebook_generation,
        "cluster_queue_name": UPGRADE_KUEUE_CLUSTER_QUEUE_NAME,
        "local_queue_name": UPGRADE_KUEUE_LOCAL_QUEUE_NAME,
        "resource_flavor_name": UPGRADE_KUEUE_RESOURCE_FLAVOR_NAME,
        "stopped_annotation_value": stopped_annotation,
    }

    ConfigMap(
        client=admin_client,
        name=UPGRADE_KUEUE_BASELINE_CM_NAME,
        namespace=UPGRADE_KUEUE_NAMESPACE,
        data={"baseline": json.dumps(baseline)},
    ).deploy()

    LOGGER.info(f"Saved kueue upgrade baseline: {baseline}")


@pytest.fixture(scope="session")
def upgrade_kueue_baseline(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
) -> dict[str, Any]:
    """Load the pre-upgrade kueue baseline from the ConfigMap.

    Returns an empty dict during pre-upgrade runs.
    """
    if not pytestconfig.option.post_upgrade:
        return {}

    cm = ConfigMap(
        client=admin_client,
        name=UPGRADE_KUEUE_BASELINE_CM_NAME,
        namespace=UPGRADE_KUEUE_NAMESPACE,
    )

    assert cm.exists, (
        f"Kueue baseline ConfigMap '{UPGRADE_KUEUE_BASELINE_CM_NAME}' not found in "
        f"namespace '{UPGRADE_KUEUE_NAMESPACE}'. Ensure pre-upgrade tests ran successfully."
    )

    cm_data = cm.instance.data or {}
    raw = cm_data.get("baseline")
    assert raw, f"Kueue baseline ConfigMap '{UPGRADE_KUEUE_BASELINE_CM_NAME}' has no 'baseline' key."

    return json.loads(raw)


@pytest.fixture(scope="session")
def new_kueue_notebook_pvc(
    unprivileged_client: DynamicClient,
    upgrade_kueue_namespace: Namespace,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    """PVC for the post-upgrade new kueue notebook."""
    with PersistentVolumeClaim(
        client=unprivileged_client,
        name=NEW_KUEUE_NOTEBOOK_NAME,
        namespace=upgrade_kueue_namespace.name,
        label={constants.Labels.OpenDataHub.DASHBOARD: "true"},
        accessmodes=PersistentVolumeClaim.AccessMode.RWO,
        size="1Gi",
        volume_mode=PersistentVolumeClaim.VolumeMode.FILE,
        teardown=True,
    ) as pvc:
        yield pvc


@pytest.fixture(scope="session")
def new_kueue_notebook(
    unprivileged_client: DynamicClient,
    upgrade_kueue_namespace: Namespace,
    upgrade_kueue_hardware_profile: HardwareProfile,
    upgrade_notebook_image: str,
    new_kueue_notebook_pvc: PersistentVolumeClaim,
) -> Generator[Notebook, Any, Any]:
    """Fresh kueue-managed Notebook CR created post-upgrade via HardwareProfile."""
    notebook_dict = build_notebook_dict(
        namespace=upgrade_kueue_namespace.name,
        name=NEW_KUEUE_NOTEBOOK_NAME,
        image_path=upgrade_notebook_image,
        extra_annotations={
            "opendatahub.io/hardware-profile-name": upgrade_kueue_hardware_profile.name,
        },
        resources={},
    )

    with Notebook(client=unprivileged_client, kind_dict=notebook_dict, teardown=True) as nb:
        yield nb


@pytest.fixture(scope="session")
def new_kueue_notebook_pod(
    unprivileged_client: DynamicClient,
    new_kueue_notebook: Notebook,
) -> Pod:
    """Pod for the post-upgrade new kueue notebook; waits for Ready state."""
    notebook_pod = Pod(
        client=unprivileged_client,
        namespace=new_kueue_notebook.namespace,
        name=f"{new_kueue_notebook.name}-0",
    )

    wait_for_notebook_pod_ready(notebook_pod=notebook_pod, context="New kueue notebook (post-upgrade)")

    return notebook_pod
