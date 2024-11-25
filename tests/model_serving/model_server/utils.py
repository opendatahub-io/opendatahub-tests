from contextlib import contextmanager
from typing import Optional, Generator, Any, Dict

from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from tests.model_serving.model_server.private_endpoint.utils import InvalidStorageArgument


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
    min_replicas: int = 1,
    wait: bool = True,
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

    with InferenceService(
        client=client,
        name=name,
        namespace=namespace,
        annotations={
            "serving.knative.openshift.io/enablePassthrough": "true",
            "sidecar.istio.io/inject": "true",
            "sidecar.istio.io/rewriteAppHTTPProbers": "true",
            "serving.kserve.io/deploymentMode": deployment_mode,
        },
        predictor=predictor_dict,
        wait_for_resource=wait,
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
