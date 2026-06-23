from typing import Generator

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.notebook import Notebook
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from pytest_testconfig import config as py_config
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutExpiredError

from tests.workbenches.notebooks_server.controller.utils import (
    build_notebook_dict,
    resolve_notebook_image,
)
from utilities import constants
from utilities.constants import Timeout
from utilities.general import collect_pod_information

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
) -> Generator[Notebook, None, None]:
    """Returns a new Notebook CR for a given namespace, name, and image"""
    namespace = request.param["namespace"]
    name = request.param["name"]

    # Optional Auth annotations
    auth_annotations = request.param.get("auth_annotations", {})

    notebook_dict = build_notebook_dict(
        namespace=namespace,
        name=name,
        image_path=notebook_image,
        extra_annotations=auth_annotations or None,
    )

    with Notebook(client=unprivileged_client, kind_dict=notebook_dict) as nb:
        yield nb


@pytest.fixture(scope="function")
def notebook_pod(
    unprivileged_client: DynamicClient,
    default_notebook: Notebook,
) -> Pod:
    """
    Returns a notebook pod in Ready state.

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
    # Error messages
    _ERR_POD_NOT_READY = (
        "Pod '{pod_name}-0' failed to reach Ready state within 10 minutes.\n"
        "Pod Phase: {pod_phase}\n"
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
            timeout=Timeout.TIMEOUT_10MIN,
        )
    except (TimeoutError, TimeoutExpiredError) as e:
        try:
            pod_exists = notebook_pod.exists
        except Exception as exists_error:  # noqa: BLE001
            LOGGER.warning(f"Failed to verify pod existence after timeout: {exists_error}")
            pod_exists = False

        if pod_exists:
            # Collect pod information for debugging purposes (YAML + logs saved to must-gather dir)
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
            # Pod was never created
            raise AssertionError(_ERR_POD_NOT_CREATED.format(pod_name=default_notebook.name)) from e

    return notebook_pod
