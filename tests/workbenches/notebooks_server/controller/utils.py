from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.pod import Pod
from ocp_resources.resource import NamespacedResource, Resource
from ocp_resources.route import Route
from ocp_resources.self_subject_review import SelfSubjectReview
from ocp_resources.user import User
from ocp_resources.service_account import ServiceAccount
from pytest_testconfig import config as py_config
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutExpiredError

from utilities.constants import Labels
from utilities.general import collect_pod_information

LOGGER = get_logger(name=__name__)

WORKBENCH_TRUSTED_CA_BUNDLE_NAME = "workbench-trusted-ca-bundle"
CA_BUNDLE_CERT_KEY = "ca-bundle.crt"
KUBEFLOW_STOPPED_ANNOTATION: str = "kubeflow-resource-stopped"


class StatefulSet(NamespacedResource):
    """StatefulSet resource (apps/v1). Not shipped by ocp_resources."""

    api_group: str = NamespacedResource.ApiGroup.APPS


class MutatingWebhookConfiguration(Resource):
    """MutatingWebhookConfiguration resource (admissionregistration.k8s.io/v1).

    Not shipped by ocp_resources.
    """

    api_group: str = Resource.ApiGroup.ADMISSIONREGISTRATION_K8S_IO


class HardwareProfile(NamespacedResource):
    """HardwareProfile resource (infrastructure.opendatahub.io/v1alpha1)."""

    api_group: str = "infrastructure.opendatahub.io"


def get_username(client: DynamicClient) -> str | None:
    """Gets the username for the client (see kubectl -v8 auth whoami)"""
    username: str | None
    try:
        self_subject_review = SelfSubjectReview(client=client, name="selfSubjectReview").create()
        assert self_subject_review
        username = self_subject_review.status.userInfo.username
    except NotImplementedError:
        LOGGER.info(
            "SelfSubjectReview not found. Falling back to user.openshift.io/v1/users/~ for OpenShift versions <=4.14"
        )
        user = User(client=client, name="~").instance
        username = user.get("metadata", {}).get("name", None)

    return username


def resolve_notebook_image(admin_client: DynamicClient) -> str:
    """Resolves the full image path for a minimal workbench notebook.

    Resolves the image from the cluster ImageStream because operator CSV versions
    (for example ``2.25``) do not always match ImageStream tag names (for example
    ``2025.2``). Using the CSV version directly causes ImagePullBackOff on
    references such as ``s2i-minimal-notebook:2.25``.

    Args:
        admin_client: Cluster client for ImageStream and product version lookups.

    Returns:
        Full image reference, preferring a digest-pinned ``dockerImageReference``.
    """
    from tests.workbenches.notebook_images.utils import (
        WorkbenchImageSpec,
        resolve_workbench_image,
    )

    imagestream_name = (
        "jupyter-minimal-notebook" if py_config.get("distribution") == "upstream" else "s2i-minimal-notebook"
    )
    return resolve_workbench_image(
        admin_client=admin_client,
        spec=WorkbenchImageSpec(
            ide="jupyterlab",
            imagestream_name=imagestream_name,
            notebook_name="workbench",
            baseline_prefix="jupyterlab",
            pvc_name="workbench-storage",
        ),
    ).image_url


@contextmanager
def notebook_service_account(
    client: DynamicClient,
    name: str,
    namespace: str,
    *,
    teardown: bool = True,
) -> Generator[ServiceAccount, Any, Any]:
    """Ensure the per-notebook ServiceAccount exists before deploying a Notebook CR.

    The Kubeflow notebook controller creates the StatefulSet immediately, but on some
    RHOAI versions the ODH controller creates auth resources asynchronously. Pre-creating
    the ServiceAccount avoids pod scheduling failures when the SA is not found.

    Args:
        client: Kubernetes client for the target namespace.
        name: ServiceAccount name (matches the notebook name).
        namespace: Target namespace.
        teardown: Whether to delete the ServiceAccount on context exit.

    Yields:
        The existing or newly created ServiceAccount.
    """
    existing_sa = ServiceAccount(client=client, name=name, namespace=namespace, ensure_exists=False)
    if existing_sa.exists:
        yield existing_sa
        return

    with ServiceAccount(client=client, name=name, namespace=namespace, teardown=teardown) as service_account:
        yield service_account


def get_dashboard_route_host(admin_client: DynamicClient) -> str:
    """Returns the hostname of the dashboard Route.

    Raises:
        ResourceNotFoundError: If the dashboard Route does not exist.
    """
    route_name = "odh-dashboard" if py_config.get("distribution") == "upstream" else "rhods-dashboard"
    route = Route(client=admin_client, name=route_name, namespace=py_config["applications_namespace"])
    if not route.exists:
        raise ResourceNotFoundError(f"Route {route.name} does not exist")
    return route.host


def build_notebook_dict(
    namespace: str,
    name: str,
    image_path: str,
    route_host: str,
    username: str,
    extra_annotations: dict[str, str] | None = None,
    extra_labels: dict[str, str] | None = None,
    resources: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Builds a Notebook CR dict for the kubeflow.org/v1 API (2.25 oauth-proxy style).

    Args:
        namespace: Target namespace for the Notebook.
        name: Notebook resource name (also used for PVC claim, service account, container).
        image_path: Full container image reference.
        route_host: Dashboard route hostname (for oauth-proxy logout URL and tornado_settings).
        username: Cluster username (for tornado_settings).
        extra_annotations: Optional annotations merged into metadata (e.g. oauth sidecar resources).
        extra_labels: Optional labels merged into metadata (e.g. kueue queue-name).
        resources: Container resources dict with "limits" and/or "requests" keys.
            None uses sensible defaults; empty dict ``{}`` omits resources entirely
            (useful when a HardwareProfile webhook injects them).

    Returns:
        A dict suitable for passing to ``Notebook(kind_dict=...)``.
    """
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

    annotations: dict[str, str] = {
        "notebooks.opendatahub.io/inject-oauth": "true",
        "opendatahub.io/accelerator-name": "",
        "opendatahub.io/service-mesh": "false",
        "notebooks.opendatahub.io/last-image-selection": image_path,
    }
    if extra_annotations:
        annotations.update(extra_annotations)

    labels: dict[str, str] = {
        Labels.Openshift.APP: name,
        Labels.OpenDataHub.DASHBOARD: "true",
        "opendatahub.io/odh-managed": "true",
        "sidecar.istio.io/inject": "false",
    }
    if extra_labels:
        labels.update(extra_labels)

    container: dict[str, Any] = {
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
                f'--ServerApp.tornado_settings={{"user":"{username}","hub_host":"https://{route_host}","hub_prefix":"/projects/{namespace}"}}',  # noqa: E501 line too long
            },
            {"name": "JUPYTER_IMAGE", "value": image_path},
        ],
        "image": image_path,
        "imagePullPolicy": "Always",
        "livenessProbe": probe_config,
        "name": name,
        "ports": [{"containerPort": 8888, "name": "notebook-port", "protocol": "TCP"}],
        "readinessProbe": probe_config,
        "volumeMounts": [
            {"mountPath": "/opt/app-root/src", "name": name},
            {"mountPath": "/dev/shm", "name": "shm"},
        ],
        "workingDir": "/opt/app-root/src",
    }

    if resources is None:
        container["resources"] = {
            "limits": {"cpu": "2", "memory": "4Gi"},
            "requests": {"cpu": "1", "memory": "1Gi"},
        }
    elif resources:
        container["resources"] = resources

    return {
        "apiVersion": "kubeflow.org/v1",
        "kind": "Notebook",
        "metadata": {
            "annotations": annotations,
            "labels": labels,
            "name": name,
            "namespace": namespace,
        },
        "spec": {
            "template": {
                "spec": {
                    "affinity": {},
                    "containers": [
                        container,
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
                                f"--logout-url=https://{route_host}/projects/{namespace}?notebookLogout={name}",
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
                    ],
                }
            }
        },
    }


def wait_for_notebook_pod_ready(notebook_pod: Pod, *, context: str, timeout: int = 300) -> None:
    """Wait for a notebook pod to reach Ready state, with diagnostic collection on failure.

    Args:
        notebook_pod: The Pod resource to wait on.
        context: Human-readable context for error messages (e.g. "Kueue notebook").
        timeout: Maximum seconds to wait for the pod to become Ready.

    Raises:
        AssertionError: If the pod does not reach Ready within *timeout* seconds
            or is never created.
    """
    try:
        notebook_pod.wait(timeout=timeout)
        notebook_pod.wait_for_condition(
            condition=Pod.Condition.READY,
            status=Pod.Condition.Status.TRUE,
            timeout=timeout,
        )
    except (TimeoutError, TimeoutExpiredError) as e:
        if notebook_pod.exists:
            collect_pod_information(notebook_pod)
            raise AssertionError(
                f"{context} pod '{notebook_pod.name}' failed to reach Ready state "
                f"within {timeout} seconds.\nOriginal error: {e}"
            ) from e

        raise AssertionError(
            f"{context} pod '{notebook_pod.name}' was not created. Check notebook controller logs."
        ) from e
