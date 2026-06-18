import json
from datetime import UTC, datetime
from typing import Any, Generator

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.namespace import Namespace
from ocp_resources.notebook import Notebook
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutExpiredError

from tests.workbenches.notebooks_server.controller.utils import (
    StatefulSet,
    build_notebook_dict,
    get_dashboard_route_host,
    get_username,
    resolve_notebook_image,
)
from utilities import constants
from utilities.constants import Timeout
from utilities.general import collect_pod_information
from utilities.infra import create_ns

LOGGER = get_logger(name=__name__)

UPGRADE_NAMESPACE = "upgrade-workbenches"
UPGRADE_NOTEBOOK_NAME = "upgrade-workbenches"
UPGRADE_STOPPED_NOTEBOOK_NAME = "upgrade-wb-stopped"
UPGRADE_BASELINE_CM_NAME = "upgrade-workbenches-baseline"


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
        route_host = get_dashboard_route_host(admin_client=admin_client)
        username = get_username(client=unprivileged_client)
        assert username, "Failed to determine username from the cluster"

        notebook_dict = build_notebook_dict(
            namespace=upgrade_notebook_namespace.name,
            name=UPGRADE_NOTEBOOK_NAME,
            image_path=upgrade_notebook_image,
            route_host=route_host,
            username=username,
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
            timeout=Timeout.TIMEOUT_5MIN,
        )
    except (TimeoutError, TimeoutExpiredError, RuntimeError) as e:
        if notebook_pod.exists:
            collect_pod_information(notebook_pod)
            raise AssertionError(
                f"Pod '{upgrade_notebook.name}-0' failed to reach Ready state "
                f"within {Timeout.TIMEOUT_5MIN} seconds.\n"
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
def upgrade_notebook_tls_service(
    unprivileged_client: DynamicClient,
    upgrade_notebook: Notebook,
) -> Service:
    """TLS Service created by the notebook controller for oauth-proxy (name-tls)."""
    return Service(
        client=unprivileged_client,
        name=f"{upgrade_notebook.name}-tls",
        namespace=upgrade_notebook.namespace,
    )


@pytest.fixture(scope="session")
def upgrade_notebook_route(
    admin_client: DynamicClient,
    upgrade_notebook: Notebook,
) -> Route:
    """OpenShift Route for the notebook (created by the notebook controller for oauth-proxy)."""
    return Route(
        client=admin_client,
        name=upgrade_notebook.name,
        namespace=upgrade_notebook.namespace,
    )


@pytest.fixture(scope="session")
def auth_oauth_config_secret(
    unprivileged_client: DynamicClient,
    upgrade_notebook: Notebook,
) -> Secret:
    """oauth-proxy config Secret for the notebook ({name}-oauth-config)."""
    return Secret(
        client=unprivileged_client,
        name=f"{upgrade_notebook.name}-oauth-config",
        namespace=upgrade_notebook.namespace,
    )


@pytest.fixture(scope="session")
def auth_tls_secret(
    unprivileged_client: DynamicClient,
    upgrade_notebook: Notebook,
) -> Secret:
    """TLS Secret for the notebook's oauth-proxy ({name}-tls)."""
    return Secret(
        client=unprivileged_client,
        name=f"{upgrade_notebook.name}-tls",
        namespace=upgrade_notebook.namespace,
    )


@pytest.fixture(scope="session")
def stopped_auth_oauth_config_secret(
    unprivileged_client: DynamicClient,
    stopped_notebook: Notebook,
) -> Secret:
    """oauth-proxy config Secret for the stopped notebook."""
    return Secret(
        client=unprivileged_client,
        name=f"{stopped_notebook.name}-oauth-config",
        namespace=stopped_notebook.namespace,
    )


@pytest.fixture(scope="session")
def stopped_auth_tls_secret(
    unprivileged_client: DynamicClient,
    stopped_notebook: Notebook,
) -> Secret:
    """TLS Secret for the stopped notebook's oauth-proxy."""
    return Secret(
        client=unprivileged_client,
        name=f"{stopped_notebook.name}-tls",
        namespace=stopped_notebook.namespace,
    )


@pytest.fixture(scope="session")
def stopped_tls_service(
    unprivileged_client: DynamicClient,
    stopped_notebook: Notebook,
) -> Service:
    """TLS Service for the stopped notebook ({name}-tls)."""
    return Service(
        client=unprivileged_client,
        name=f"{stopped_notebook.name}-tls",
        namespace=stopped_notebook.namespace,
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
        route_host = get_dashboard_route_host(admin_client=admin_client)
        username = get_username(client=unprivileged_client)
        assert username, "Failed to determine username from the cluster"

        notebook_dict = build_notebook_dict(
            namespace=upgrade_notebook_namespace.name,
            name=UPGRADE_STOPPED_NOTEBOOK_NAME,
            image_path=upgrade_notebook_image,
            route_host=route_host,
            username=username,
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
            timeout=Timeout.TIMEOUT_5MIN,
        )
    except (TimeoutError, TimeoutExpiredError, RuntimeError) as e:
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

    notebook_pod.wait_deleted(timeout=Timeout.TIMEOUT_2MIN)
    LOGGER.info(f"Pod '{notebook_pod.name}' terminated after stop annotation")

    replicas = stopped_notebook_statefulset.instance.spec.replicas
    assert replicas == 0, (
        f"StatefulSet '{stopped_notebook_statefulset.name}' has {replicas} replicas after stop, expected 0"
    )
    LOGGER.info(f"StatefulSet '{stopped_notebook_statefulset.name}' confirmed at 0 replicas")


@pytest.fixture(scope="session")
def capture_notebook_baseline(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    upgrade_notebook: Notebook,
    upgrade_notebook_pod: Pod,
    upgrade_notebook_statefulset: StatefulSet,
    upgrade_notebook_service: Service,
    upgrade_notebook_tls_service: Service,
    upgrade_notebook_route: Route,
    stopped_notebook: Notebook,
    stopped_notebook_pre_upgrade_shutdown: None,
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

    assert upgrade_notebook_tls_service.exists, (
        f"TLS Service '{upgrade_notebook_tls_service.name}' not found in "
        f"'{upgrade_notebook_tls_service.namespace}' during baseline capture"
    )
    tls_service_spec = upgrade_notebook_tls_service.instance.spec
    tls_service_ports = json.dumps(tls_service_spec.ports, sort_keys=True, default=str)
    tls_service_selector = json.dumps(tls_service_spec.selector, sort_keys=True, default=str)

    assert upgrade_notebook_route.exists, (
        f"Route '{upgrade_notebook_route.name}' not found in "
        f"'{upgrade_notebook_route.namespace}' during baseline capture"
    )
    route_generation = upgrade_notebook_route.instance.metadata.generation
    route_host = upgrade_notebook_route.host
    route_tls_termination = upgrade_notebook_route.instance.spec.get("tls", {}).get("termination", "")

    stopped_annotation = stopped_notebook.instance.metadata.annotations.get("kubeflow-resource-stopped")

    baseline = {
        "ntb_creation_timestamp": creation_timestamp,
        "notebook_generation": notebook_generation,
        "statefulset_generation": sts_generation,
        "service_ports": service_ports,
        "service_selector": service_selector,
        "tls_service_ports": tls_service_ports,
        "tls_service_selector": tls_service_selector,
        "route_generation": route_generation,
        "route_host": route_host,
        "route_tls_termination": route_tls_termination,
        "stopped_annotation_value": stopped_annotation,
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
