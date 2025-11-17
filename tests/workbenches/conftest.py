from typing import Generator

import pytest
from pytest_testconfig import config as py_config

from simple_logger.logger import get_logger
from tests.workbenches.utils import get_username

from kubernetes.dynamic import DynamicClient

from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.notebook import Notebook

from utilities.constants import Labels
from utilities import constants
from utilities.constants import INTERNAL_IMAGE_REGISTRY_PATH
from utilities.infra import check_internal_image_registry_available

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
    """Provides a full image name of a minimal workbench image"""
    image_name = "jupyter-minimal-notebook" if py_config.get("distribution") == "upstream" else "s2i-minimal-notebook"
    yield f"{image_name}:{'2025.2'}"


@pytest.fixture(scope="function")
def default_notebook(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    minimal_image: str,
) -> Generator[Notebook, None, None]:
    """Returns a new Notebook CR for a given namespace, name, and image"""
    namespace = request.param["namespace"]
    name = request.param["name"]

    # Optional Auth annotations
    auth_annotations = request.param.get("auth_annotations", {})

    # Optional custom image parameter (for custom workbench image testing)
    custom_image = request.param.get("custom_image", None)

    # Validate custom_image if provided
    if custom_image is not None:
        custom_image = custom_image.strip()
        if not custom_image:
            raise ValueError("custom_image cannot be empty or whitespace")

    # Set the correct username
    username = get_username(dyn_client=admin_client)
    assert username, "Failed to determine username from the cluster"

    # Error messages
    _ERR_INVALID_CUSTOM_IMAGE = (
        "custom_image must be a valid OCI image reference "
        "(e.g., 'quay.io/org/image:tag' or 'quay.io/org/image@sha256:digest'), "
        "got: '{custom_image}'"
    )

    # Determine which image to use
    if custom_image:
        # Custom image provided - use it directly (must be valid OCI image reference)
        if ":" not in custom_image or "/repository/" in custom_image or "/manifest/" in custom_image:
            raise ValueError(_ERR_INVALID_CUSTOM_IMAGE.format(custom_image=custom_image))
        minimal_image_path = custom_image
        LOGGER.info(f"Using custom workbench image: {custom_image}")
    else:
        # No custom image - use default minimal image with registry resolution
        # Check internal image registry availability
        internal_image_registry = check_internal_image_registry_available(admin_client=admin_client)

        # Set the image path based on internal image registry status
        minimal_image_path = (
            f"{INTERNAL_IMAGE_REGISTRY_PATH}/{py_config['applications_namespace']}/{minimal_image}"
            if internal_image_registry
            else ":" + minimal_image.rsplit(":", maxsplit=1)[1]
        )

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
                "notebooks.opendatahub.io/last-image-selection": minimal_image_path if custom_image else minimal_image,
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
                                {"name": "JUPYTER_IMAGE", "value": minimal_image_path},
                            ],
                            "image": minimal_image_path,
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
