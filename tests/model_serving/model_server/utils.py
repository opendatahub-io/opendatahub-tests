import base64
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from kubernetes.dynamic.exceptions import ResourceNotFoundError

from tests.model_serving.model_server.private_endpoint.utils import (
    InvalidStorageArgument,
)
from utilities.constants import KServeDeploymentType


@contextmanager
def create_isvc(
    client: DynamicClient,
    name: str,
    namespace: str,
    deployment_mode: str,
    model_format: str,
    runtime: str,
    storage_uri: Optional[str] = None,
    storage_key: Optional[str] = None,
    storage_path: Optional[str] = None,
    wait: bool = True,
    enable_auth: bool = False,
    external_route: bool = False,
    model_service_account: Optional[str] = "",
    min_replicas: Optional[int] = None,
    argument: Optional[list[str]] = None,
    resources: Optional[dict[str, Any]] = None,
    volumes: Optional[dict[str, Any]] = None,
    volumes_mounts: Optional[dict[str, Any]] = None,
) -> Generator[InferenceService, Any, Any]:
    predictor_dict: Dict[str, Any] = {
        "minReplicas": min_replicas,
        "model": {
            "modelFormat": {"name": model_format},
            "version": "1",
            "runtime": runtime,
        },
    }

    _check_storage_arguments(storage_uri, storage_key, storage_path)
    if storage_uri:
        predictor_dict["model"]["storageUri"] = storage_uri
    elif storage_key:
        predictor_dict["model"]["storage"] = {"key": storage_key, "path": storage_path}
    if model_service_account:
        predictor_dict["serviceAccountName"] = model_service_account

    if min_replicas:
        predictor_dict["minReplicas"] = min_replicas
    if argument:
        predictor_dict["model"]["args"] = argument
    if resources:
        predictor_dict["model"]["resources"] = resources
    if volumes_mounts:
        predictor_dict["model"]["volumeMounts"] = volumes_mounts
    if volumes:
        predictor_dict["volumes"] = volumes
    annotations = {"serving.kserve.io/deploymentMode": deployment_mode}

    if deployment_mode == KServeDeploymentType.SERVERLESS:
        annotations.update({
            "serving.knative.openshift.io/enablePassthrough": "true",
            "sidecar.istio.io/inject": "true",
            "sidecar.istio.io/rewriteAppHTTPProbers": "true",
        })

    if enable_auth:
        annotations["security.opendatahub.io/enable-auth"] = "true"

    if external_route:
        annotations["networking.kserve.io/visibility"] = "exposed"

    with InferenceService(
        client=client,
        name=name,
        namespace=namespace,
        annotations=annotations,
        predictor=predictor_dict,
    ) as inference_service:
        if wait:
            inference_service.wait_for_condition(
                condition=inference_service.Condition.READY,
                status=inference_service.Condition.Status.TRUE,
                timeout=10 * 60,
            )
        yield inference_service


def _check_storage_arguments(
    storage_uri: Optional[str],
    storage_key: Optional[str],
    storage_path: Optional[str],
) -> None:
    if (storage_uri and storage_path) or (not storage_uri and not storage_key) or (storage_key and not storage_path):
        raise InvalidStorageArgument(storage_uri, storage_key, storage_path)


def b64_encoded_string(string_to_encode: str) -> str:
    """Returns openshift compliant base64 encoding of a string

    encodes the input string to bytes-like, encodes the bytes-like to base 64,
    decodes the b64 to a string and returns it. This is needed for openshift
    resources expecting b64 encoded values in the yaml.

    Args:
        string_to_encode: The string to encode in base64

    Returns:
        A base64 encoded string that is compliant with openshift's yaml format
    """
    return base64.b64encode(string_to_encode.encode()).decode()


def get_pods_by_isvc_label(client: DynamicClient, isvc: InferenceService) -> List[Pod]:
    """
    Args:
        client (DynamicClient): OCP Client to use.
        isvc (InferenceService):InferenceService object.

    Returns:
        list[Pod]: A list of all matching pods

    Raises:
        ResourceNotFoundError: if no pods are found.
    """
    if pods := [
        pod
        for pod in Pod.get(
            dyn_client=client,
            namespace=isvc.namespace,
            label_selector=f"{isvc.ApiGroup.SERVING_KSERVE_IO}/inferenceservice={isvc.name}",
        )
    ]:
        return pods

    raise ResourceNotFoundError(f"{isvc.name} has no pods")
