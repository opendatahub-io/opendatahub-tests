from typing import Generator

import pytest
from pytest_testconfig import config as py_config

from simple_logger.logger import get_logger
from tests.workbenches.utils import get_username

from kubernetes.dynamic import DynamicClient

from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.notebook import Notebook
from ocp_resources.pod import Pod

from utilities.constants import Labels, Timeout
from utilities import constants
from utilities.constants import INTERNAL_IMAGE_REGISTRY_PATH
from utilities.infra import get_product_version
from utilities.infra import check_internal_image_registry_available
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
def minimal_image(admin_client: DynamicClient) -> Generator[str, None, None]:
    """Provides a full image name of a minimal workbench image."""
    image_name = "jupyter-minimal-notebook" if py_config.get("distribution") == "upstream" else "s2i-minimal-notebook"
    image_tag = py_config.get("workbench_image_tag")

    if not image_tag:
        product_version = get_product_version(admin_client=admin_client)
        image_tag = f"{product_version.major}.{product_version.minor}"

    yield f"{image_name}:{image_tag}"


@pytest.fixture(scope="function")
def notebook_image(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    minimal_image: str,
) -> str:
    """
    Resolves the notebook image path.

    Priority:
    1. 'custom_image' provided via indirect parametrization
    2. Default 'minimal_image' (with automatic registry resolution)
    """
    # SAFELY get parameters. If test doesn't parameterize this fixture, default to empty dict.
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
    internal_image_registry = check_internal_image_registry_available(admin_client=admin_client)

    return (
        f"{INTERNAL_IMAGE_REGISTRY_PATH}/{py_config['applications_namespace']}/{minimal_image}"
        if internal_image_registry
        else minimal_image
    )


@pytest.fixture(scope="function")
def default_notebook(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    notebook_image: str,
) -> Generator[Notebook, None, None]:
    """Returns a new Notebook CR for a given namespace, name, and image"""
    namespace = request.param["namespace"]
    name = request.param["name"]

    # Optional Auth annotations
    auth_annotations = request.param.get("auth_annotations", {})

    # Set the correct username
    username = get_username(client=admin_client)
    assert username, "Failed to determine username from the cluster"

    # Set the image path based on the resolved notebook_image
    image_path = notebook_image

    probe_config = {
        "failureThreshold": 3,
        "httpGet": {
            "path": f"/notebook/{namespace}/{name}/api",
            "port": "notebook-port",
            "scheme": "HTTP",
        },
        "initialDelaySeconds": 10,
        "periodSeconds": 5,
        "successThreshold": 1,
        "timeoutSeconds": 1,
    }

    notebook = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "Notebook",
        "metadata": {
            "annotations": {
                Labels.Notebook.INJECT_AUTH: "true",
                "opendatahub.io/accelerator-name": "",
                "notebooks.opendatahub.io/last-image-selection": image_path,
                # Add any additional annotations if provided
                **auth_annotations,
            },
            "finalizers": [
                "notebook.opendatahub.io/kube-rbac-proxy-cleanup",
            ],
            "labels": {
                Labels.Openshift.APP: name,
                Labels.OpenDataHub.DASHBOARD: "true",
                "opendatahub.io/odh-managed": "true",
            },
            "name": name,
            "namespace": namespace,
        },
        "spec": {
            "template": {
                "spec": {
                    "affinity": {},
                    "containers": [
                        {
                            "env": [
                                {
                                    "name": "NOTEBOOK_ARGS",
                                    "value": "--ServerApp.port=8888\n"
                                    "                  "
                                    "--ServerApp.token=''\n"
                                    "                  "
                                    "--ServerApp.password=''\n"
                                    "                  "
                                    f"--ServerApp.base_url=/notebook/{namespace}/{name}\n"
                                    "                  "
                                    "--ServerApp.quit_button=False\n",
                                },
                                {"name": "JUPYTER_IMAGE", "value": image_path},
                            ],
                            "image": image_path,
                            "imagePullPolicy": "Always",
                            "livenessProbe": probe_config,
                            "name": name,
                            "ports": [{"containerPort": 8888, "name": "notebook-port", "protocol": "TCP"}],
                            "readinessProbe": probe_config,
                            "resources": {
                                "limits": {"cpu": "2", "memory": "4Gi"},
                                "requests": {"cpu": "1", "memory": "1Gi"},
                            },
                            "volumeMounts": [
                                {"mountPath": "/opt/app-root/src", "name": name},
                                {"mountPath": "/dev/shm", "name": "shm"},
                            ],
                            "workingDir": "/opt/app-root/src",
                        },
                    ],
                    "enableServiceLinks": False,
                    "serviceAccountName": name,
                    "volumes": [
                        {"name": name, "persistentVolumeClaim": {"claimName": name}},
                        {"emptyDir": {"medium": "Memory"}, "name": "shm"},
                        {
                            "name": "kube-rbac-proxy-config",
                            "configMap": {"defaultMode": 420, "name": "test-kube-rbac-proxy-config"},
                        },
                        {
                            "name": "kube-rbac-proxy-tls-certificates",
                            "secret": {
                                "defaultMode": 420,
                                "secretName": "test-kube-rbac-proxy-tls",  # pragma: allowlist secret
                            },
                        },
                    ],
                }
            }
        },
    }

    with Notebook(kind_dict=notebook) as nb:
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
    except (TimeoutError, RuntimeError) as e:
        if notebook_pod.exists:
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
