from typing import Any

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod
from ocp_resources.resource import NamespacedResource, Resource
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutExpiredError

from utilities.constants import INTERNAL_IMAGE_REGISTRY_PATH, Labels
from utilities.general import collect_pod_information
from utilities.infra import check_internal_image_registry_available, get_product_version

LOGGER = structlog.get_logger(name=__name__)

WORKBENCH_TRUSTED_CA_BUNDLE_NAME: str = "workbench-trusted-ca-bundle"
CA_BUNDLE_CERT_KEY: str = "ca-bundle.crt"
KUBEFLOW_STOPPED_ANNOTATION: str = "kubeflow-resource-stopped"


class StatefulSet(NamespacedResource):
    """StatefulSet resource (apps/v1). Not shipped by ocp_resources."""

    api_group: str = NamespacedResource.ApiGroup.APPS


class MutatingWebhookConfiguration(Resource):
    """MutatingWebhookConfiguration resource (admissionregistration.k8s.io/v1)."""

    api_group: str = Resource.ApiGroup.ADMISSIONREGISTRATION_K8S_IO


class HardwareProfile(NamespacedResource):
    """HardwareProfile resource (infrastructure.opendatahub.io/v1)."""

    api_group: str = "infrastructure.opendatahub.io"


def resolve_notebook_image(admin_client: DynamicClient) -> str:
    """Resolves the full image path for a minimal workbench notebook.

    Determines the image name based on distribution (upstream/downstream),
    resolves the tag from config or product version, and prepends the
    internal registry path when available.

    Args:
        admin_client: Cluster client for querying product version and registry availability.

    Returns:
        Full image reference (e.g. "image-registry.../namespace/jupyter-minimal-notebook:2.18").
    """
    image_name = "jupyter-minimal-notebook" if py_config.get("distribution") == "upstream" else "s2i-minimal-notebook"
    image_tag = py_config.get("workbench_image_tag")

    if not image_tag:
        product_version = get_product_version(admin_client=admin_client)
        image_tag = f"{product_version.major}.{product_version.minor}"

    minimal_image = f"{image_name}:{image_tag}"
    internal_image_registry = check_internal_image_registry_available(admin_client=admin_client)

    return (
        f"{INTERNAL_IMAGE_REGISTRY_PATH}/{py_config['applications_namespace']}/{minimal_image}"
        if internal_image_registry
        else minimal_image
    )


def build_notebook_dict(
    namespace: str,
    name: str,
    image_path: str,
    extra_annotations: dict[str, str] | None = None,
    extra_labels: dict[str, str] | None = None,
    resources: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Builds a Notebook CR dict for the kubeflow.org/v1 API.

    Args:
        namespace: Target namespace for the Notebook.
        name: Notebook resource name (also used for PVC claim, service account, container).
        image_path: Full container image reference.
        extra_annotations: Optional annotations merged into metadata (e.g. auth sidecar resources).
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

    container_resources: dict[str, dict[str, str]] | None = (
        resources
        if resources is not None
        else {
            "limits": {"cpu": "2", "memory": "4Gi"},
            "requests": {"cpu": "1", "memory": "1Gi"},
        }
    )

    annotations: dict[str, str] = {
        Labels.Notebook.INJECT_AUTH: "true",
        "opendatahub.io/accelerator-name": "",
        "notebooks.opendatahub.io/last-image-selection": image_path,
    }
    if extra_annotations:
        annotations.update(extra_annotations)

    labels: dict[str, str] = {
        Labels.Openshift.APP: name,
        Labels.OpenDataHub.DASHBOARD: "true",
        "opendatahub.io/odh-managed": "true",
    }
    if extra_labels:
        labels.update(extra_labels)

    return {
        "apiVersion": "kubeflow.org/v1",
        "kind": "Notebook",
        "metadata": {
            "annotations": annotations,
            "finalizers": [
                "notebook.opendatahub.io/kube-rbac-proxy-cleanup",
            ],
            "labels": labels,
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
                            **({"resources": container_resources} if container_resources else {}),
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
