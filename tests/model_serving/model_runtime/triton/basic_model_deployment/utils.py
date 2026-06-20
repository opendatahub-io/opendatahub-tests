import json
from typing import Any

import portforward
import requests
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_runtime.triton.constant import (
    ACCELERATOR_IDENTIFIER,
    LOCAL_HOST_URL,
    TEMPLATE_MAP,
    TRITON_REST_PORT,
)
from utilities.constants import KServeDeploymentType, Labels, RuntimeTemplates, Timeout


def send_rest_request(url: str, input_data: dict[str, Any]) -> Any:
    response = requests.post(url=url, json=input_data, verify=False, timeout=Timeout.TIMEOUT_3MIN)
    response.raise_for_status()
    return response.json()


def run_triton_inference(
    pod_name: str, isvc: InferenceService, input_data: dict[str, Any], model_name: str
) -> Any:
    deployment_mode = isvc.instance.metadata.annotations.get("serving.kserve.io/deploymentMode")
    rest_endpoint = f"/v2/models/{model_name}/infer"

    supported_modes = (KServeDeploymentType.RAW_DEPLOYMENT, KServeDeploymentType.STANDARD)
    if deployment_mode in supported_modes:
        with portforward.forward(
            pod_or_service=pod_name, namespace=isvc.namespace, from_port=TRITON_REST_PORT, to_port=TRITON_REST_PORT
        ):
            host = f"{LOCAL_HOST_URL}:{TRITON_REST_PORT}"
            return send_rest_request(f"{host}{rest_endpoint}", input_data)

    return f"Invalid deployment_mode {deployment_mode}"


def validate_inference_request(
    pod_name: str,
    isvc: InferenceService,
    response_snapshot: Any,
    input_query: Any,
    model_name: str,
) -> None:
    response = run_triton_inference(
        pod_name=pod_name,
        isvc=isvc,
        input_data=input_query,
        model_name=model_name,
    )

    assert response, "Response is empty"
    assert isinstance(response, dict), f"Response is not a dict: {response}"
    assert response.get("outputs"), "Response missing outputs"

    if "rawOutputContents" in response or "raw_output_contents" in response:
        raw_contents = response.get("rawOutputContents") or response.get("raw_output_contents")
        assert raw_contents
        return

    assert isinstance(response["outputs"], list), "Outputs must be a list"
    assert len(response["outputs"]) > 0, "Outputs list is empty"

    output = response["outputs"][0]
    assert isinstance(output, dict), f"Output must be a dict, got {type(output).__name__}"

    actual_data = output.get("data", [])
    assert actual_data, "Data is empty"
    assert isinstance(actual_data, list), f"Data must be a list, got {type(actual_data).__name__}"

    top_k = min(5, len(actual_data))
    actual_top_k = sorted(range(len(actual_data)), key=lambda i: actual_data[i], reverse=True)[:top_k]
    assert all(isinstance(i, int) and 0 <= i < len(actual_data) for i in actual_top_k)


def get_gpu_identifier(accelerator_type: str | None) -> str:
    if accelerator_type is None:
        return Labels.Nvidia.NVIDIA_COM_GPU
    return ACCELERATOR_IDENTIFIER.get(accelerator_type.lower(), Labels.Nvidia.NVIDIA_COM_GPU)


def get_template_name(accelerator_type: str | None) -> str:
    if accelerator_type is None:
        accelerator_type = "nvidia"
    key = f"rest_{accelerator_type.lower()}"
    return TEMPLATE_MAP.get(key, RuntimeTemplates.TRITON_REST)


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)
