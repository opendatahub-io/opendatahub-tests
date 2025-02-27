from __future__ import annotations

from pytest_testconfig import config as py_config

from kubernetes.dynamic import DynamicClient, Resource, ResourceInstance

from ocp_resources.route import Route
from ocp_resources.notebook import Notebook

from utilities.constants import INTERNAL_IMAGE_REGISTRY_PATH


def get_notebook_image(image_name: str, image_tag: str) -> str:
    controllers_namespace = py_config["applications_namespace"]
    if py_config.get("distribution") == "upstream":
        image_dict = {"jupyter-minimal-notebook": "jupyter-minimal-notebook"}
    else:
        image_dict = {"jupyter-minimal-notebook": "s2i-minimal-notebook"}
    return f"{INTERNAL_IMAGE_REGISTRY_PATH}/{controllers_namespace}/{image_dict[image_name]}:{image_tag}"


def load_default_notebook(dyn_client: DynamicClient, namespace: str, name: str, image: str) -> Notebook:
    # Set new Route url
    route_name = "odh-dashboard" if py_config.get("distribution") == "upstream" else "rhods-dashboard"
    route = Route(client=dyn_client, name=route_name, namespace=py_config["applications_namespace"])
    assert route.exists, f"Route {route.name} does not exist"

    # Set the correct username
    username = get_username(dyn_client=dyn_client)

    notebook = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "Notebook",
        "metadata": {
            "annotations": {
                "notebooks.opendatahub.io/inject-oauth": "true",
                "opendatahub.io/accelerator-name": "",
                "opendatahub.io/service-mesh": "false",
            },
            "labels": {
                "app": name,
                "opendatahub.io/dashboard": "true",
                "opendatahub.io/odh-managed": "true",
                "sidecar.istio.io/inject": "false",
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
                                {"name": "JUPYTER_IMAGE", "value": image},
                            ],
                            "image": image,
                            "imagePullPolicy": "Always",
                            "livenessProbe": {
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
                            },
                            "name": name,
                            "ports": [{"containerPort": 8888, "name": "notebook-port", "protocol": "TCP"}],
                            "readinessProbe": {
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
                            },
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

    return Notebook(kind_dict=notebook)


def get_username(dyn_client: DynamicClient) -> str:
    """Gets the username for the client (see kubectl -v8 auth whoami)"""
    self_subject_review_resource: Resource = dyn_client.resources.get(
        api_version="authentication.k8s.io/v1", kind="SelfSubjectReview"
    )
    self_subject_review: ResourceInstance = dyn_client.create(self_subject_review_resource)
    username: str = self_subject_review.status.userInfo.username
    return username
