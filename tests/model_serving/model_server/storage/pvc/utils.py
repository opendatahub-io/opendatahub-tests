from contextlib import contextmanager
from typing import Optional, Generator, Any

from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from tests.model_serving.model_server.private_endpoint.utils import MissingStorageArgument


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
) -> Generator[InferenceService,Any,Any]:
    predictor_storage={
            "minReplicas": min_replicas,
            "model": {
                "modelFormat": {"name": model_format},
                "version": "1",
                "runtime": runtime,
            },
        }
    if storage_uri:
        predictor_storage["model"]["storageUri"] = storage_uri
    elif storage_key and storage_path:
        predictor_storage["storage"] =  {
                "key": storage_key,
                "path": storage_path,
            }
    else:
        raise MissingStorageArgument(storage_uri, storage_key, storage_path)

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
        predictor=predictor_storage,
    ) as inference_service:
        if wait:
            inference_service.wait_for_condition(
                condition=inference_service.Condition.READY,
                status=inference_service.Condition.Status.TRUE,
                timeout=10 * 60,
            )
        yield inference_service
