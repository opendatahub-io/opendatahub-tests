from contextlib import contextmanager
from typing import Optional, Generator, Any

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
    predictor_dict = {
        "minReplicas": min_replicas,
        "model": {
            "modelFormat": {"name": model_format},
            "version": "1",
            "runtime": runtime,
        },
    }
    # "type:ignore" is needed because otherwise mypy will complain about "Unsupported target for indexed assignment ("object")"
    if storage_uri and storage_key and storage_path:
        raise InvalidStorageArgument(storage_uri, storage_key, storage_path)
    elif storage_uri:
        predictor_dict["model"]["storageUri"] = storage_uri  # type: ignore
    elif storage_key and storage_path:
        predictor_dict["model"]["storage"] = {"key": storage_key, "path": storage_path}  # type: ignore
    else:
        raise InvalidStorageArgument(storage_uri, storage_key, storage_path)

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
