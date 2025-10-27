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

    # Set the correct username
    username = get_username(dyn_client=admin_client)
    assert username, "Failed to determine username from the cluster"

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
                "notebooks.opendatahub.io/last-image-selection": minimal_image,
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
                        {
                            "resources": {
                                "limits": {"cpu": "100m", "memory": "64Mi"},
                                "requests": {"cpu": "100m", "memory": "64Mi"},
                            },
                            "readinessProbe": {
                                "failureThreshold": 3,
                                "httpGet": {"path": "/healthz", "port": 8444, "scheme": "HTTPS"},
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5,
                                "successThreshold": 1,
                                "timeoutSeconds": 1,
                            },
                            "name": "kube-rbac-proxy",
                            "livenessProbe": {
                                "failureThreshold": 3,
                                "httpGet": {"path": "/healthz", "port": 8444, "scheme": "HTTPS"},
                                "initialDelaySeconds": 30,
                                "periodSeconds": 5,
                                "successThreshold": 1,
                                "timeoutSeconds": 1,
                            },
                            "ports": [{"containerPort": 8443, "name": "kube-rbac-proxy", "protocol": "TCP"}],
                            "imagePullPolicy": "Always",
                            "volumeMounts": [
                                {"mountPath": "/etc/kube-rbac-proxy", "name": "kube-rbac-proxy-config"},
                                {"mountPath": "/etc/tls/private", "name": "kube-rbac-proxy-tls-certificates"},
                            ],
                            "image": "quay.io/rhoai/odh-kube-auth-proxy-rhel9:rhoai-3.0",
                            "args": [
                                "--secure-listen-address=0.0.0.0:8443",
                                "--upstream=http://127.0.0.1:8888/",
                                "--logtostderr=true",
                                "--v=10",
                                "--proxy-endpoints-port=8444",
                                "--config-file=/etc/kube-rbac-proxy/config-file.yaml",
                                "--tls-cert-file=/etc/tls/private/tls.crt",
                                "--tls-private-key-file=/etc/tls/private/tls.key",
                                "--auth-header-fields-enabled=true",
                                "--auth-header-user-field-name=X-Auth-Request-User",
                                "--auth-header-groups-field-name=X-Auth-Request-Groups",
                            ],
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
