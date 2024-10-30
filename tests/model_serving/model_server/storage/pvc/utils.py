from contextlib import contextmanager
from typing import Any, Dict, Optional

from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService

@contextmanager
def create_isvc(
        client: DynamicClient,
        name: str,
        namespace: str,
        deployment_mode: str,
        storage_key: str,
        storage_uri: str,
        model_format: str,
        runtime: str,
        max_replicas: Optional[int] = 1,
        min_replicas: Optional[int] = 1,
        cpu_limit: str = "2",
        memory_limit: str = "8Gi",
        cpu_request: str = "1",
        memory_request: str = "4Gi",
        wait: bool = True,
        enable_auth: bool = False,
        model_service_account: Optional[str] = ""
) -> InferenceService:


    predictor_config: Dict[str, Any] = {
        "model": {
            "modelFormat": {"name": model_format},
            "version": "1",
            "runtime": runtime,
            "storage": {
                "key": storage_key,
                "path": storage_uri
            },
            "resources": {
                "limits": {
                    "cpu": cpu_limit,
                    "memory": memory_limit
                },
                "requests": {
                    "cpu": cpu_request,
                    "memory": memory_request
                }
            }
        },
        "maxReplicas": max_replicas,
        "minReplicas": min_replicas
    }

    if model_service_account:
        predictor_config["serviceAccountName"] = model_service_account

    if deployment_mode == "RawDeployment":
        annotations = {
            "serving.kserve.io/deploymentMode": deployment_mode,
        }
        labels = {
            "networking.kserve.io/visibility": "enable-route",
            "security.opendatahub.io/enable-auth": "true",
        }
    else:
        annotations = {
            "serving.knative.openshift.io/enablePassthrough": "true",
            "sidecar.istio.io/inject": "true",
            "sidecar.istio.io/rewriteAppHTTPProbers": "true",
            "serving.kserve.io/deploymentMode": deployment_mode,
        }

    if enable_auth and deployment_mode == "Serverless":
        annotations["security.opendatahub.io/enable-auth"] = "true"

    with InferenceService(
            client=client,
            name=name,
            namespace=namespace,
            label=labels,
            annotations=annotations,
            predictor=predictor_config,
    ) as inference_service:
        if wait:
            inference_service.wait_for_condition(
                condition=inference_service.Condition.READY,
                status=inference_service.Condition.Status.TRUE,
                timeout=10 * 60,
            )

        yield inference_service
