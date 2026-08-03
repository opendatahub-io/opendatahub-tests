from typing import Any, Generator

import pytest
import yaml
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.config_map import ConfigMap
from ocp_resources.namespace import Namespace
from ocp_resources.notebook import Notebook
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from pytest_testconfig import config as py_config
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.workbenches.notebooks_server.controller.utils import (
    HardwareProfile,
    build_notebook_dict,
    get_dashboard_route_host,
    get_username,
    resolve_notebook_image,
)
from utilities import constants
from utilities.constants import Timeout
from utilities.general import collect_pod_information
from utilities.infra import create_ns
from utilities.kueue_utils import (
    ClusterQueue,
    Kueue,
    LocalQueue,
    ResourceFlavor,
    create_cluster_queue,
    create_local_queue,
    create_resource_flavor,
)

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="function")
def users_persistent_volume_claim(
    request: pytest.FixtureRequest, unprivileged_model_namespace: Namespace, unprivileged_client: DynamicClient
) -> Generator[PersistentVolumeClaim, None, None]:
    with PersistentVolumeClaim(
        client=unprivileged_client,
        name=request.param["name"],
        namespace=unprivileged_model_namespace.name,
        label={constants.Labels.OpenDataHub.DASHBOARD: "true"},
        accessmodes=PersistentVolumeClaim.AccessMode.RWO,
        size="10Gi",
        volume_mode=PersistentVolumeClaim.VolumeMode.FILE,
    ) as pvc:
        yield pvc


@pytest.fixture(scope="function")
def minimal_image() -> Generator[str, None, None]:
    """Provides a full image name of a minimal workbench image (name:tag only, no registry prefix)."""
    image_name = "jupyter-minimal-notebook" if py_config.get("distribution") == "upstream" else "s2i-minimal-notebook"
    image_tag = py_config.get("workbench_image_tag", "2025.2")

    yield f"{image_name}:{image_tag}"


@pytest.fixture(scope="function")
def default_notebook(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
    minimal_image: str,
) -> Generator[Notebook, None, None]:
    """Returns a new Notebook CR for a given namespace, name, and image"""
    namespace = request.param["namespace"]
    name = request.param["name"]

    oauth_annotations = request.param.get("oauth_annotations", {})

    route_host = get_dashboard_route_host(admin_client=admin_client)

    username = get_username(client=unprivileged_client)
    assert username, "Failed to determine username from the cluster"

    image_path = resolve_notebook_image(admin_client=admin_client)

    notebook_dict = build_notebook_dict(
        namespace=namespace,
        name=name,
        image_path=image_path,
        route_host=route_host,
        username=username,
        extra_annotations=oauth_annotations or None,
    )

    with Notebook(kind_dict=notebook_dict) as nb:
        yield nb


@pytest.fixture(scope="function")
def notebook_pod(
    unprivileged_client: DynamicClient,
    default_notebook: Notebook,
) -> Pod:
    """Returns a notebook pod in Ready state.

    This fixture:
    - Creates a Pod object for the notebook
    - Waits for pod to exist
    - Waits for pod to reach Ready state (10-minute timeout)
    - Provides detailed diagnostics on failure

    Args:
        unprivileged_client: Client for interacting with the cluster
        default_notebook: The notebook CR to get the pod for

    Returns:
        Pod object in Ready state

    Raises:
        AssertionError: If pod fails to reach Ready state or is not created
    """
    _ERR_POD_NOT_READY = (
        "Pod '{pod_name}-0' failed to reach Ready state within 10 minutes.\n"
        "Pod Phase: {pod_phase}\n"
        "Original Error: {original_error}\n"
        "Pod information collected to must-gather directory for debugging."
    )
    _ERR_POD_NOT_CREATED = "Pod '{pod_name}-0' was not created. Check notebook controller logs."

    notebook_pod = Pod(
        client=unprivileged_client,
        namespace=default_notebook.namespace,
        name=f"{default_notebook.name}-0",
    )

    try:
        notebook_pod.wait()
        notebook_pod.wait_for_condition(
            condition=Pod.Condition.READY,
            status=Pod.Condition.Status.TRUE,
            timeout=Timeout.TIMEOUT_10MIN,
        )
    except (TimeoutError, TimeoutExpiredError) as e:
        try:
            pod_exists = notebook_pod.exists
        except Exception as exists_error:  # noqa: BLE001
            LOGGER.warning(f"Failed to verify pod existence after timeout: {exists_error}")
            pod_exists = False

        if pod_exists:
            collect_pod_information(notebook_pod)
            pod_status = notebook_pod.instance.status
            pod_phase = pod_status.phase
            raise AssertionError(
                _ERR_POD_NOT_READY.format(
                    pod_name=default_notebook.name,
                    pod_phase=pod_phase,
                    original_error=e,
                )
            ) from e
        else:
            raise AssertionError(_ERR_POD_NOT_CREATED.format(pod_name=default_notebook.name)) from e

    return notebook_pod


# ---------------------------------------------------------------------------
# Kueue Integration Fixtures
# ---------------------------------------------------------------------------


KUEUE_CONTROLLER_NAMESPACE: str = "redhat-ods-applications"
KUEUE_MANAGER_CONFIG_CM: str = "kueue-manager-config"
KUEUE_CONTROLLER_LABEL: str = "app.kubernetes.io/name=kueue"


def _restart_kueue_controller(admin_client: DynamicClient) -> None:
    """Delete the Kueue controller pod and wait for a new one to become Ready."""
    pods = list(
        Pod.get(
            dyn_client=admin_client,
            namespace=KUEUE_CONTROLLER_NAMESPACE,
            label_selector=KUEUE_CONTROLLER_LABEL,
        )
    )
    for pod in pods:
        pod.delete(wait=True)

    for sample in TimeoutSampler(
        wait_timeout=120,
        sleep=5,
        func=lambda: list(
            Pod.get(
                dyn_client=admin_client,
                namespace=KUEUE_CONTROLLER_NAMESPACE,
                label_selector=KUEUE_CONTROLLER_LABEL,
            )
        ),
    ):
        if sample and all(
            pod.status == Pod.Status.RUNNING
            and any(
                c.get("type") == "Ready" and c.get("status") == "True" for c in (pod.instance.status.conditions or [])
            )
            for pod in sample
        ):
            break


@pytest.fixture(scope="session")
def kueue_statefulset_framework_check(admin_client: DynamicClient) -> Generator[None, None, None]:
    """Ensure Kueue has StatefulSet framework enabled for notebook scheduling.

    Notebooks are backed by StatefulSets, so Kueue must have 'pod' and 'statefulset'
    listed in integrations.frameworks for Workload objects to be created.

    For unmanaged mode (Red Hat build of Kueue operator), checks the Kueue CR.
    For managed mode (embedded in RHOAI), reads the kueue-manager-config ConfigMap
    and patches it if statefulset is missing, then restores it on teardown.
    """
    # Try unmanaged mode first (Red Hat build of Kueue operator with Kueue CR)
    try:
        kueue_cr = Kueue(
            client=admin_client,
            name="cluster",
        )
        if kueue_cr.exists:
            spec = kueue_cr.instance.to_dict().get("spec", {})
            frameworks: list[str] = spec.get("config", {}).get("integrations", {}).get("frameworks", [])
            assert "StatefulSet" in frameworks, (
                f"Kueue CR 'cluster' does not have 'StatefulSet' in config.integrations.frameworks. "
                f"Current frameworks: {frameworks}. "
                f"Notebooks require StatefulSet integration. "
                f"Patch the Kueue CR to add 'StatefulSet' to spec.config.integrations.frameworks."
            )
            yield
            return
    except (ResourceNotFoundError, TimeoutExpiredError, NotImplementedError):
        LOGGER.info("Kueue CR not found via kueue.openshift.io — checking managed/embedded mode")

    # Managed/embedded mode: read the kueue-manager-config ConfigMap
    cm = ConfigMap(
        client=admin_client,
        name=KUEUE_MANAGER_CONFIG_CM,
        namespace=KUEUE_CONTROLLER_NAMESPACE,
    )
    assert cm.exists, (
        f"ConfigMap '{KUEUE_MANAGER_CONFIG_CM}' not found in namespace "
        f"'{KUEUE_CONTROLLER_NAMESPACE}'. Cannot verify Kueue configuration."
    )

    config_yaml_str = cm.instance.data["controller_manager_config.yaml"]
    config_data = yaml.safe_load(config_yaml_str)
    integrations: dict[str, Any] = config_data.setdefault("integrations", {})
    frameworks_list: list[str] = integrations.get("frameworks", [])

    needs_pod = "pod" not in frameworks_list
    needs_statefulset = "statefulset" not in frameworks_list

    if not needs_pod and not needs_statefulset:
        LOGGER.info("Kueue config already has pod+statefulset frameworks enabled")
        yield
        return

    # Patch: add missing frameworks
    original_config_yaml = config_yaml_str
    if needs_pod:
        frameworks_list.append("pod")
    if needs_statefulset:
        frameworks_list.append("statefulset")
    integrations["frameworks"] = frameworks_list

    # Add podOptions.namespaceSelector to avoid managing system namespaces
    if "podOptions" not in integrations:
        integrations["podOptions"] = {
            "namespaceSelector": {
                "matchExpressions": [
                    {
                        "key": "kubernetes.io/metadata.name",
                        "operator": "NotIn",
                        "values": ["kube-system", "kueue-system", KUEUE_CONTROLLER_NAMESPACE],
                    }
                ]
            }
        }

    new_config_yaml = yaml.dump(config_data, default_flow_style=False)
    LOGGER.info(
        f"Patching Kueue config to add frameworks: "
        f"{'pod ' if needs_pod else ''}{'statefulset ' if needs_statefulset else ''}"
    )

    # Set opendatahub.io/managed=false to prevent the RHOAI operator from
    # reconciling the ConfigMap back to its original state.
    cm_body = cm.instance.to_dict()
    annotations = cm_body["metadata"].setdefault("annotations", {})
    original_managed_value = annotations.get("opendatahub.io/managed")
    annotations["opendatahub.io/managed"] = "false"
    cm_body["data"]["controller_manager_config.yaml"] = new_config_yaml
    cm.update_replace(resource_dict=cm_body)

    _restart_kueue_controller(admin_client=admin_client)
    LOGGER.info("Kueue controller restarted with statefulset framework enabled")

    yield

    # Restore original config and remove the opt-out annotation
    LOGGER.info("Restoring original Kueue config (removing statefulset framework)")
    cm_body = cm.instance.to_dict()
    cm_body["data"]["controller_manager_config.yaml"] = original_config_yaml
    if original_managed_value is not None:
        cm_body["metadata"]["annotations"]["opendatahub.io/managed"] = original_managed_value
    else:
        cm_body["metadata"]["annotations"].pop("opendatahub.io/managed", None)
    cm.update_replace(resource_dict=cm_body)
    _restart_kueue_controller(admin_client=admin_client)
    LOGGER.info("Kueue controller restored to original configuration")


@pytest.fixture(scope="class")
def kueue_notebook_namespace(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
) -> Generator[Namespace, None, None]:
    """Namespace with kueue.openshift.io/managed=true label for kueue workload management."""
    with create_ns(
        admin_client=admin_client,
        name=request.param["name"],
        unprivileged_client=unprivileged_client,
        add_kueue_label=True,
    ) as ns:
        yield ns


@pytest.fixture(scope="class")
def kueue_resource_flavor(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
) -> Generator[ResourceFlavor, None, None]:
    """ResourceFlavor for kueue notebook workloads."""
    with create_resource_flavor(
        client=admin_client,
        name=request.param["name"],
    ) as resource_flavor:
        yield resource_flavor


@pytest.fixture(scope="class")
def kueue_cluster_queue(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    kueue_resource_flavor: ResourceFlavor,
) -> Generator[ClusterQueue, None, None]:
    """ClusterQueue with CPU/memory quotas for notebook workloads."""
    resource_groups = [
        {
            "coveredResources": ["cpu", "memory"],
            "flavors": [
                {
                    "name": kueue_resource_flavor.name,
                    "resources": [
                        {"name": "cpu", "nominalQuota": request.param["cpu_quota"]},
                        {"name": "memory", "nominalQuota": request.param["memory_quota"]},
                    ],
                }
            ],
        }
    ]

    with create_cluster_queue(
        client=admin_client,
        name=request.param["name"],
        resource_groups=resource_groups,
        namespace_selector=request.param.get("namespace_selector", {}),
    ) as cluster_queue:
        yield cluster_queue


@pytest.fixture(scope="class")
def kueue_local_queue(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    kueue_notebook_namespace: Namespace,
    kueue_cluster_queue: ClusterQueue,
) -> Generator[LocalQueue, None, None]:
    """LocalQueue in the kueue-enabled namespace, bound to the ClusterQueue."""
    with create_local_queue(
        client=admin_client,
        name=request.param["name"],
        cluster_queue=kueue_cluster_queue.name,
        namespace=kueue_notebook_namespace.name,
    ) as local_queue:
        yield local_queue


@pytest.fixture(scope="class")
def kueue_notebook_pvc(
    request: pytest.FixtureRequest,
    unprivileged_client: DynamicClient,
    kueue_notebook_namespace: Namespace,
) -> Generator[PersistentVolumeClaim, None, None]:
    """PVC for notebook storage in the kueue-enabled namespace."""
    with PersistentVolumeClaim(
        client=unprivileged_client,
        name=request.param["name"],
        namespace=kueue_notebook_namespace.name,
        label={constants.Labels.OpenDataHub.DASHBOARD: "true"},
        accessmodes=PersistentVolumeClaim.AccessMode.RWO,
        size="10Gi",
        volume_mode=PersistentVolumeClaim.VolumeMode.FILE,
    ) as pvc:
        yield pvc


@pytest.fixture(scope="class")
def kueue_hardware_profile(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    kueue_notebook_namespace: Namespace,
    kueue_local_queue: LocalQueue,
) -> Generator[HardwareProfile, None, None]:
    """HardwareProfile with scheduling.type=Queue for Kueue-backed workbenches."""
    hwp_dict = {
        "apiVersion": "infrastructure.opendatahub.io/v1alpha1",
        "kind": "HardwareProfile",
        "metadata": {
            "name": request.param["name"],
            "namespace": kueue_notebook_namespace.name,
        },
        "spec": {
            "identifiers": [
                {
                    "displayName": "CPU",
                    "identifier": "cpu",
                    "minCount": "100m",
                    "maxCount": request.param.get("cpu_max", "4"),
                    "defaultCount": request.param["cpu_default"],
                    "resourceType": "CPU",
                },
                {
                    "displayName": "Memory",
                    "identifier": "memory",
                    "minCount": "128Mi",
                    "maxCount": request.param.get("memory_max", "8Gi"),
                    "defaultCount": request.param["memory_default"],
                    "resourceType": "Memory",
                },
            ],
            "scheduling": {
                "type": "Queue",
                "kueue": {
                    "localQueueName": kueue_local_queue.name,
                },
            },
        },
    }

    with HardwareProfile(client=admin_client, kind_dict=hwp_dict) as hwp:
        yield hwp


@pytest.fixture(scope="class")
def kueue_notebook(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
    kueue_notebook_namespace: Namespace,
    kueue_notebook_pvc: PersistentVolumeClaim,
    kueue_hardware_profile: HardwareProfile,
) -> Generator[Notebook, None, None]:
    """Notebook CR annotated with a HardwareProfile for Kueue scheduling.

    The HWP webhook injects the kueue.x-k8s.io/queue-name label and container
    resources (from HWP identifiers.defaultCount) into the Notebook CR.
    """
    notebook_image = resolve_notebook_image(admin_client=admin_client)
    route_host = get_dashboard_route_host(admin_client=admin_client)
    username = get_username(client=unprivileged_client)
    assert username, "Failed to determine username from the cluster"

    notebook_dict = build_notebook_dict(
        namespace=kueue_notebook_namespace.name,
        name=request.param["name"],
        image_path=notebook_image,
        route_host=route_host,
        username=username,
        extra_annotations={"opendatahub.io/hardware-profile-name": kueue_hardware_profile.name},
        resources={},
    )

    with Notebook(client=unprivileged_client, kind_dict=notebook_dict) as nb:
        yield nb
