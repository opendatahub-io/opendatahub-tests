from typing import Generator

import pytest
from pytest_testconfig import config as py_config

from simple_logger.logger import get_logger
from tests.workbenches.utils import get_username

from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError

from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.route import Route
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

    # Optional OAuth annotations
    oauth_annotations = request.param.get("oauth_annotations", {})

    # Set new Route url
    route_name = "odh-dashboard" if py_config.get("distribution") == "upstream" else "rhods-dashboard"
    route = Route(client=admin_client, name=route_name, namespace=py_config["applications_namespace"])
    if not route.exists:
        raise ResourceNotFoundError(f"Route {route.name} does not exist")

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
                "notebooks.opendatahub.io/inject-oauth": "true",
                "opendatahub.io/accelerator-name": "",
                "opendatahub.io/service-mesh": "false",
                "notebooks.opendatahub.io/last-image-selection": minimal_image,
                # Add any additional annotations if provided
                **oauth_annotations,
            },
            "labels": {
                Labels.Openshift.APP: name,
                Labels.OpenDataHub.DASHBOARD: "true",
                "opendatahub.io/odh-managed": "true",
                "sidecar.istio.io/inject": "false",
                **request.param.get("labels", {}),  # Merge custom labels from parametrize
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
                                    "--ServerApp.quit_button=False\n"
                                    "                  "
                                    f'--ServerApp.tornado_settings={{"user":"{username}","hub_host":"https://{route.host}","hub_prefix":"/projects/{namespace}"}}',  # noqa: E501 line too long
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
                                "requests": {
                                    "cpu": request.param.get("cpu_request", "1"),
                                    "memory": request.param.get("memory_request", "1Gi"),
                                },
                            },
                            "volumeMounts": [
                                {"mountPath": "/opt/app-root/src", "name": name},
                                {"mountPath": "/dev/shm", "name": "shm"},
                            ],
                            "workingDir": "/opt/app-root/src",
                        },
                        {
                            "args": [
                                "--provider=openshift",
                                "--https-address=:8443",
                                "--http-address=",
                                f"--openshift-service-account={name}",
                                "--cookie-secret-file=/etc/oauth/config/cookie_secret",
                                "--cookie-expire=24h0m0s",
                                "--tls-cert=/etc/tls/private/tls.crt",
                                "--tls-key=/etc/tls/private/tls.key",
                                "--upstream=http://localhost:8888",
                                "--upstream-ca=/var/run/secrets/kubernetes.io/serviceaccount/ca.crt",
                                "--email-domain=*",
                                "--skip-provider-button",
                                f'--openshift-sar={{"verb":"get","resource":"notebooks","resourceAPIGroup":"kubeflow.org","resourceName":"{name}","namespace":"$(NAMESPACE)"}}',  # noqa: E501 line too long
                                f"--logout-url=https://{route.host}/projects/{namespace}?notebookLogout={name}",
                            ],
                            "env": [
                                {"name": "NAMESPACE", "valueFrom": {"fieldRef": {"fieldPath": "metadata.namespace"}}}
                            ],
                            "image": "registry.redhat.io/openshift4/ose-oauth-proxy:v4.10",
                            "imagePullPolicy": "Always",
                            "livenessProbe": {
                                "failureThreshold": 3,
                                "httpGet": {"path": "/oauth/healthz", "port": "oauth-proxy", "scheme": "HTTPS"},
                                "initialDelaySeconds": 30,
                                "periodSeconds": 5,
                                "successThreshold": 1,
                                "timeoutSeconds": 1,
                            },
                            "name": "oauth-proxy",
                            "ports": [{"containerPort": 8443, "name": "oauth-proxy", "protocol": "TCP"}],
                            "readinessProbe": {
                                "failureThreshold": 3,
                                "httpGet": {"path": "/oauth/healthz", "port": "oauth-proxy", "scheme": "HTTPS"},
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5,
                                "successThreshold": 1,
                                "timeoutSeconds": 1,
                            },
                            "resources": {
                                "limits": {"cpu": "100m", "memory": "64Mi"},
                                "requests": {"cpu": "100m", "memory": "64Mi"},
                            },
                            "volumeMounts": [
                                {"mountPath": "/etc/oauth/config", "name": "oauth-config"},
                                {"mountPath": "/etc/tls/private", "name": "tls-certificates"},
                            ],
                        },
                    ],
                    "enableServiceLinks": False,
                    "serviceAccountName": name,
                    "volumes": [
                        {"name": name, "persistentVolumeClaim": {"claimName": name}},
                        {"emptyDir": {"medium": "Memory"}, "name": "shm"},
                        {
                            "name": "oauth-config",
                            "secret": {"defaultMode": 420, "secretName": f"{name}-oauth-config"},
                        },
                        {"name": "tls-certificates", "secret": {"defaultMode": 420, "secretName": f"{name}-tls"}},
                    ],
                }
            }
        },
    }

    with Notebook(kind_dict=notebook) as nb:
        yield nb
