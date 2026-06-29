from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.resource import NamespacedResource, Resource
from ocp_resources.service_account import ServiceAccount
from pytest_testconfig import config as py_config

from utilities.constants import Labels

LOGGER = structlog.get_logger(name=__name__)

WORKBENCH_TRUSTED_CA_BUNDLE_NAME = "workbench-trusted-ca-bundle"
CA_BUNDLE_CERT_KEY = "ca-bundle.crt"


class StatefulSet(NamespacedResource):
    """StatefulSet resource (apps/v1). Not shipped by ocp_resources."""

    api_group: str = NamespacedResource.ApiGroup.APPS


class MutatingWebhookConfiguration(Resource):
    """MutatingWebhookConfiguration resource (admissionregistration.k8s.io/v1)."""

    api_group: str = Resource.ApiGroup.ADMISSIONREGISTRATION_K8S_IO


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


def build_notebook_dict(
    namespace: str,
    name: str,
    image_path: str,
    extra_annotations: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Builds a Notebook CR dict for the kubeflow.org/v1 API.

    Args:
        namespace: Target namespace for the Notebook.
        name: Notebook resource name (also used for PVC claim, service account, container).
        image_path: Full container image reference.
        extra_annotations: Optional annotations merged into metadata (e.g. auth sidecar resources).

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
        Labels.Notebook.INJECT_AUTH: "true",
        "opendatahub.io/accelerator-name": "",
        "notebooks.opendatahub.io/last-image-selection": image_path,
    }
    if extra_annotations:
        annotations.update(extra_annotations)

    return {
        "apiVersion": "kubeflow.org/v1",
        "kind": "Notebook",
        "metadata": {
            "annotations": annotations,
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
                    ],
                }
            }
        },
    }
