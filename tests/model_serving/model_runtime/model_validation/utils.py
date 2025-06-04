import pytest
import requests
import portforward
from simple_logger.logger import get_logger
import json 
from contextlib import contextmanager
from typing import Generator, Any
from kubernetes.dynamic import DynamicClient
from ocp_resources.secret import Secret
from ocp_resources.inference_service import InferenceService
from utilities.constants import KServeDeploymentType, Protocols
from tests.model_serving.model_runtime.model_validation.constant import PULL_SECRET_ACCESS_TYPE, COMPLETION_QUERY



LOGGER = get_logger(name=__name__)


@contextmanager
def kserve_registry_pull_secret(
    admin_client: DynamicClient,
    name: str,
    namespace: str,
    registry_pull_secret: str,
    registry_host: str,
) -> Generator[Secret, Any, Any]:
    with Secret(
        client=admin_client,
        name=name,
        namespace=namespace,
        string_data={
            ".dockerconfigjson": registry_pull_secret,
            "ACCESS_TYPE": PULL_SECRET_ACCESS_TYPE,
            "OCI_HOST": registry_host,
        },
        wait_for_resource=True,
    ) as secret:
        yield secret


def send_rest_request(
    url: str,
    input_data: dict
) -> Any:
    response = requests.post(url=url, json=input_data, verify=False, timeout=60)
    response.raise_for_status()
    return response.json()




def run_modelcar_inference(
    pod_name: str,
    isvc: InferenceService,
    input_data: dict,
    protocol: str,
) -> Any:
    """
    Run inference against a Modelcar model using Vllm.
    Handles SERVERLESS deployment mode.
    """
    deployment_mode = isvc.instance.metadata.annotations.get("serving.kserve.io/deploymentMode")
    model_name = isvc.instance.metadata.name
    
    if protocol not in (Protocols.HTTP, Protocols.HTTPS):
        return f"Invalid protocol {protocol}"
        
    if deployment_mode == KServeDeploymentType.SERVERLESS:
        base_url = isvc.instance.status.url.rstrip('/')
        completion_endpoint = "/v1/completions"
        model_name = isvc.instance.metadata.name
        # Prepare the request payload
        payload = {
            "model": model_name,
            "prompt": input_data.get("prompt", ""),
            "max_tokens": input_data.get("max_tokens", 256)
        }
        
        # Add any additional parameters from input_data
        for key, value in input_data.items():
            if key not in ["prompt", "max_tokens"]:
                payload[key] = value
                
        return send_rest_request(f"{base_url}{completion_endpoint}", payload)
        
    return f"Invalid deployment_mode {deployment_mode}"


def validate_inference_request(
    pod_name: str,
    isvc: InferenceService,
    response_snapshot: Any,
    input_query: Any,
    protocol: str,
) -> None:
    response = run_modelcar_inference(
        pod_name=pod_name,
        isvc=isvc,
        input_data=input_query,
        protocol=protocol,
        )
    assert response == response_snapshot, f"Output mismatch: {response} != {response_snapshot}"