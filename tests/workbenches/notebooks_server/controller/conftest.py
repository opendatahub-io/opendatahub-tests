from collections.abc import Generator
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.notebook import Notebook
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutExpiredError

from tests.workbenches.notebooks_server.controller.utils import (
    HardwareProfile,
    build_notebook_dict,
    resolve_notebook_image,
)
from utilities import constants
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

LOGGER = structlog.get_logger(name=__name__)


def _read_obj_field(obj: Any, field_name: str, default: Any = None) -> Any:
    """Safely read attribute/key from k8s dynamic objects."""
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    return getattr(obj, field_name, default)


def _format_container_status(container_status: Any, status_prefix: str) -> str:
    """Build one-line diagnostic summary for a container status."""
    name = _read_obj_field(obj=container_status, field_name="name", default="<unknown>")
    ready = _read_obj_field(obj=container_status, field_name="ready", default=False)
    restart_count = _read_obj_field(obj=container_status, field_name="restartCount", default=0)
    state = _read_obj_field(obj=container_status, field_name="state", default={})

    state_description = "unknown"
    details: list[str] = []
    for state_name in ("waiting", "terminated", "running"):
        state_value = _read_obj_field(obj=state, field_name=state_name, default=None)
        if not state_value:
            continue

        reason = _read_obj_field(obj=state_value, field_name="reason", default=None)
        message = _read_obj_field(obj=state_value, field_name="message", default=None)
        state_description = state_name if not reason else f"{state_name}({reason})"
        if message:
            details.append(f"message={message}")
        break

    details_str = f", {', '.join(details)}" if details else ""
    return f"{status_prefix} '{name}': ready={ready}, restarts={restart_count}, state={state_description}{details_str}"


def _collect_notebook_pod_diagnostics(notebook_pod: Pod) -> str:
    """Collect concise pod status details for pytest assertion messages."""
    pod_instance = notebook_pod.instance
    pod_status = _read_obj_field(obj=pod_instance, field_name="status", default=None)
    pod_phase = _read_obj_field(obj=pod_status, field_name="phase", default="Unknown")
    pod_reason = _read_obj_field(obj=pod_status, field_name="reason", default=None)
    pod_message = _read_obj_field(obj=pod_status, field_name="message", default=None)

    lines = [f"Pod phase={pod_phase}, reason={pod_reason}, message={pod_message}"]

    pod_conditions = _read_obj_field(obj=pod_status, field_name="conditions", default=[]) or []
    lines.extend(
        "Condition "
        f"{_read_obj_field(obj=condition, field_name='type', default='<unknown>')}: "
        f"status={_read_obj_field(obj=condition, field_name='status', default='Unknown')}, "
        f"reason={_read_obj_field(obj=condition, field_name='reason', default='')}, "
        f"message={_read_obj_field(obj=condition, field_name='message', default='')}"
        for condition in pod_conditions
    )

    init_container_statuses = _read_obj_field(obj=pod_status, field_name="initContainerStatuses", default=[]) or []
    lines.extend(
        _format_container_status(container_status=container_status, status_prefix="Init container")
        for container_status in init_container_statuses
    )

    container_statuses = _read_obj_field(obj=pod_status, field_name="containerStatuses", default=[]) or []
    lines.extend(
        _format_container_status(container_status=container_status, status_prefix="Container")
        for container_status in container_statuses
    )

    return "\n".join(lines)


@pytest.fixture(scope="function")
def users_persistent_volume_claim(
    request: pytest.FixtureRequest, unprivileged_model_namespace: Namespace, unprivileged_client: DynamicClient
) -> Generator[PersistentVolumeClaim]:
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
def minimal_image(admin_client: DynamicClient) -> Generator[str]:
    """Provides a full image name of a minimal workbench image (name:tag only, no registry prefix)."""
    image_name = "jupyter-minimal-notebook" if py_config.get("distribution") == "upstream" else "s2i-minimal-notebook"
    image_tag = py_config.get("workbench_image_tag")

    if not image_tag:
        from utilities.infra import get_product_version

        product_version = get_product_version(admin_client=admin_client)
        image_tag = f"{product_version.major}.{product_version.minor}"

    yield f"{image_name}:{image_tag}"


@pytest.fixture(scope="function")
def notebook_image(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    minimal_image: str,
) -> str:
    """Resolves the notebook image path.

    Priority:
    1. 'custom_image' provided via indirect parametrization
    2. Default minimal image via ``resolve_notebook_image()``.
    """
    params = getattr(request, "param", {})
    custom_image = params.get("custom_image")

    # Case A: Custom Image (Explicit)
    if custom_image:
        custom_image = custom_image.strip()
        if not custom_image:
            raise ValueError("custom_image cannot be empty or whitespace")

        # Validation Logic: Only digest references are accepted
        _ERR_INVALID_CUSTOM_IMAGE = (
            "custom_image must be a valid OCI image reference with a digest (@sha256:digest), "
            "e.g., 'quay.io/org/image@sha256:abc123...', "
            "got: '{custom_image}'"
        )
        # Check for valid digest: @sha256: must be followed by non-empty content
        digest_marker = "@sha256:"
        has_valid_digest = False
        if digest_marker in custom_image:
            digest_index = custom_image.rfind(digest_marker)
            digest_end = digest_index + len(digest_marker)
            has_valid_digest = digest_end < len(custom_image)

        if not has_valid_digest:
            raise ValueError(_ERR_INVALID_CUSTOM_IMAGE.format(custom_image=custom_image))

        LOGGER.info(f"Using custom workbench image: {custom_image}")
        return custom_image

    # Case B: Default Image (Implicit / Good Default)
    # This runs for all standard tests in test_spawning.py
    return resolve_notebook_image(admin_client=admin_client)


@pytest.fixture(scope="function")
def default_notebook(
    request: pytest.FixtureRequest,
    unprivileged_client: DynamicClient,
    notebook_image: str,
    users_persistent_volume_claim: PersistentVolumeClaim,
) -> Generator[Notebook]:
    """Returns a new Notebook CR for a given namespace, name, and image.

    The PVC fixture dependency guarantees the Notebook is created only after
    the user PVC exists, avoiding pod scheduling races on claim lookup.
    """
    namespace = request.param["namespace"]
    name = request.param["name"]

    # Optional Auth annotations
    auth_annotations = request.param.get("auth_annotations", {})

    # Optional custom resource requests/limits for the notebook container
    custom_resources = request.param.get("resources")

    # Optional extra environment variables for the notebook container
    extra_env_vars = request.param.get("extra_env_vars")

    notebook_dict = build_notebook_dict(
        namespace=namespace,
        name=name,
        image_path=notebook_image,
        extra_annotations=auth_annotations or None,
        resources=custom_resources,
        extra_env_vars=extra_env_vars,
    )

    with Notebook(client=unprivileged_client, kind_dict=notebook_dict) as nb:
        yield nb


@pytest.fixture(scope="function")
def notebook_pod(
    request: pytest.FixtureRequest,
    unprivileged_client: DynamicClient,
    default_notebook: Notebook,
) -> Pod:
    """
    Returns a notebook pod in Ready state.

    This fixture:
    - Creates a Pod object for the notebook
    - Waits for pod to exist
    - Waits for pod to reach Ready state (configurable timeout)
    - Provides detailed diagnostics on failure

    Args:
        request: Optional fixture params. Supports {"timeout": <seconds>} via indirect parametrization.
        unprivileged_client: Client for interacting with the cluster
        default_notebook: The notebook CR to get the pod for

    Returns:
        Pod object in Ready state

    Raises:
        AssertionError: If pod fails to reach Ready state or is not created
    """
    params = getattr(request, "param", {})
    pod_ready_timeout = params.get("timeout", 600)

    # Error messages
    _ERR_POD_NOT_READY = (
        "Pod '{pod_name}-0' failed to reach Ready state within {timeout_seconds} seconds.\n"
        "Pod diagnostics:\n{pod_diagnostics}\n"
        "Original Error: {original_error}\n"
        "Pod information collected to must-gather directory for debugging."
    )
    _ERR_POD_NOT_CREATED = "Pod '{pod_name}-0' was not created. Check notebook controller logs."

    # Create pod object
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
            timeout=pod_ready_timeout,
        )
    except (TimeoutError, TimeoutExpiredError) as e:
        try:
            pod_exists = notebook_pod.exists
        except Exception as exists_error:  # noqa: BLE001
            LOGGER.warning(f"Failed to verify pod existence after timeout: {exists_error}")
            pod_exists = False

        if pod_exists:
            # Collect pod information for debugging purposes (YAML + logs saved to must-gather dir)
            try:
                collect_pod_information(notebook_pod)
            except Exception as collect_error:  # noqa: BLE001
                LOGGER.warning(f"Failed to collect pod artifacts: {collect_error}")

            try:
                pod_diagnostics = _collect_notebook_pod_diagnostics(notebook_pod=notebook_pod)
            except Exception as diagnostics_error:  # noqa: BLE001
                pod_diagnostics = f"<failed to collect pod diagnostics: {diagnostics_error}>"

            raise AssertionError(
                _ERR_POD_NOT_READY.format(
                    pod_name=default_notebook.name,
                    timeout_seconds=pod_ready_timeout,
                    pod_diagnostics=pod_diagnostics,
                    original_error=e,
                )
            ) from e
        else:
            # Pod was never created
            raise AssertionError(_ERR_POD_NOT_CREATED.format(pod_name=default_notebook.name)) from e

    return notebook_pod


# ---------------------------------------------------------------------------
# Kueue Integration Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def kueue_statefulset_framework_check(admin_client: DynamicClient) -> None:
    """Verify that the Kueue CR has StatefulSet framework enabled.

    Notebooks are backed by StatefulSets, so the Red Hat build of Kueue operator
    must have 'StatefulSet' listed in config.integrations.frameworks for workbench
    scheduling to work. Fails the test with a clear message if misconfigured.
    """
    kueue_cr = Kueue(
        client=admin_client,
        name="cluster",
        ensure_exists=True,
    )
    spec = kueue_cr.instance.to_dict().get("spec", {})
    frameworks: list[str] = spec.get("config", {}).get("integrations", {}).get("frameworks", [])

    assert "StatefulSet" in frameworks, (
        f"Kueue CR 'cluster' does not have 'StatefulSet' in config.integrations.frameworks. "
        f"Current frameworks: {frameworks}. "
        f"Notebooks require StatefulSet integration. "
        f"Patch the Kueue CR to add 'StatefulSet' to spec.config.integrations.frameworks."
    )


@pytest.fixture(scope="class")
def kueue_notebook_namespace(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
) -> Generator[Namespace]:
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
) -> Generator[ResourceFlavor]:
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
) -> Generator[ClusterQueue]:
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
) -> Generator[LocalQueue]:
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
) -> Generator[PersistentVolumeClaim]:
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
) -> Generator[HardwareProfile]:
    """HardwareProfile with scheduling.type=Queue for Kueue-backed workbenches."""
    hwp_dict = {
        "apiVersion": "infrastructure.opendatahub.io/v1",
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
) -> Generator[Notebook]:
    """Notebook CR annotated with a HardwareProfile for Kueue scheduling.

    The HWP webhook injects the kueue.x-k8s.io/queue-name label and container
    resources (from HWP identifiers.defaultCount) into the Notebook CR.
    """
    notebook_image = resolve_notebook_image(admin_client=admin_client)
    notebook_dict = build_notebook_dict(
        namespace=kueue_notebook_namespace.name,
        name=request.param["name"],
        image_path=notebook_image,
        extra_annotations={"opendatahub.io/hardware-profile-name": kueue_hardware_profile.name},
        resources={},
    )

    with Notebook(client=unprivileged_client, kind_dict=notebook_dict) as nb:
        yield nb
