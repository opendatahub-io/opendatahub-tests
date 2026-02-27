"""
Utility functions for OpenVINO model serving tests.

This module provides functions for:
- Managing S3 secrets for model access
- Sending inference requests via REST protocols
- Running inference against OpenVINO deployments
- Validating responses against snapshots
"""

import json
import os
from typing import Any

import portforward
import requests
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_runtime.openvino.constant import (
    BASE_RAW_DEPLOYMENT_CONFIG,
    LOCAL_HOST_URL,
    MODEL_PATH_PREFIX,
    OPENVINO_REST_PORT,
    RAW_DEPLOYMENT_TYPE,
)
from utilities.constants import KServeDeploymentType


def send_rest_request(url: str, input_data: dict[str, Any], verify: bool = True) -> Any:
    """
    Sends a REST POST request to the specified URL with the given JSON payload.

    Args:
        url (str): The endpoint URL to send the request to.
        input_data (dict[str, Any]): The input payload to send as JSON.
        verify (bool): TLS certificate verification (True by default).

    Returns:
        Any: The parsed JSON response from the server.

    Raises:
        requests.HTTPError: If the response contains an HTTP error status.
    """
    response = requests.post(url=url, json=input_data, verify=verify, timeout=60)
    response.raise_for_status()
    return response.json()


def run_openvino_inference(
    pod_name: str,
    isvc: InferenceService,
    input_data: dict[str, Any],
    model_version: str,
) -> Any:
    """
    Run inference against an OpenVINO-hosted model using REST protocol.
    Supports RAW KServe deployment mode.

    Args:
        pod_name (str): Name of the pod running the OpenVINO model (used for RAW deployment).
        isvc (InferenceService): The KServe InferenceService object.
        input_data (dict[str, Any]): The input data payload for inference.
        model_version (str): The version of the model to target, if applicable.

    Returns:
        Any: The inference result from the model, or an error message string.

    Notes:
        - REST calls expect the model to support V2 REST inference APIs.
        - RAW deployments use port-forwarding.
    """
    annotations = getattr(isvc.instance.metadata, "annotations", {}) or {}
    deployment_mode = annotations.get("serving.kserve.io/deploymentMode")
    if not deployment_mode:
        raise ValueError("Missing 'serving.kserve.io/deploymentMode' annotation on InferenceService")

    model_name = isvc.instance.metadata.name
    version_suffix = f"/versions/{model_version}" if model_version else ""
    rest_endpoint = f"/v2/models/{model_name}{version_suffix}/infer"

    if deployment_mode == KServeDeploymentType.RAW_DEPLOYMENT:
        port = OPENVINO_REST_PORT
        with portforward.forward(pod_or_service=pod_name, namespace=isvc.namespace, from_port=port, to_port=port):
            host = f"{LOCAL_HOST_URL}:{port}"
            return send_rest_request(url=f"{host}{rest_endpoint}", input_data=input_data, verify=False)

    raise ValueError(f"Invalid deployment_mode {deployment_mode}")


def validate_inference_request(
    pod_name: str,
    isvc: InferenceService,
    response_snapshot: Any,
    input_query: Any,
    model_version: str,
) -> None:
    """
    Runs an inference request against an OpenVINO model and validates
    that the response matches the expected snapshot.

    Args:
        pod_name (str): The pod name where the model is running.
        isvc (InferenceService): The KServe InferenceService instance.
        response_snapshot (Any): The expected inference output to compare against.
        input_query (Any): The input data to send to the model.
        model_version (str): The version of the model to target.
        root_dir (str): The root directory containing protobuf files for gRPC.

    Raises:
        AssertionError: If the actual response does not match the snapshot.
    """

    response = run_openvino_inference(
        pod_name=pod_name,
        isvc=isvc,
        input_data=input_query,
        model_version=model_version,
    )

    assert response == response_snapshot, f"Output mismatch: {response} != {response_snapshot}"


def get_model_storage_uri_dict(model_format_name: str) -> dict[str, str]:
    """
    Generate a dictionary containing the storage path for a given model format.

    This utility helps build a consistent storage URI dictionary, typically used
    for referencing model directories in file systems or remote storage.

    Args:
        model_format_name (str): Name of the model format or subdirectory.

    Returns:
        dict[str, str]: A dictionary with a single key "model-dir" pointing to the
                        constructed path using the global MODEL_PATH_PREFIX.
                        Example: {"model-dir": "/mnt/models/sklearn"}
    """
    return {"model-dir": f"{MODEL_PATH_PREFIX.rstrip('/')}/{model_format_name.lstrip('/')}"}


def get_model_namespace_dict(model_format_name: str, deployment_type: str, protocol_type: str) -> dict[str, str]:
    """
    Generate a dictionary containing a unique model namespace or name identifier.

    The function constructs a name by concatenating the given model format,
    deployment type, and protocol type using hyphens. It is useful for dynamically
    naming model-serving resources, configurations, or deployments.

    Args:
        model_format_name (str): The model format name (e.g., "onnx").
        deployment_type (str): The type of deployment (e.g., "raw").
        protocol_type (str): The communication protocol (e.g., "rest").

    Returns:
        dict[str, str]: A dictionary with the key "name" and a concatenated identifier as value.
                        Example: {"name": "onnx-raw-rest"}
    """
    name = f"{model_format_name.strip()}-{deployment_type.strip()}-{protocol_type.strip()}"
    return {"name": name}


def get_deployment_config_dict(model_format_name: str, deployment_type: str, gpu_count: int = 0) -> dict[str, str]:
    """
    Generate a deployment configuration dictionary based on the model format and deployment type.

    This function merges a base deployment configuration (raw) with a given model format
    name to produce a complete configuration dictionary.

    Args:
        model_format_name (str): The model format name (e.g., "onnx").
        deployment_type (str): The deployment type (e.g., "raw").
        gpu_count (int): The number of GPUs to allocate (default: 0).

    Returns:
        dict[str, str]: A dictionary containing the deployment configuration.
    """
    deployment_config_dict = {}

    if deployment_type == RAW_DEPLOYMENT_TYPE:
        deployment_config_dict = {"name": model_format_name, "gpu_count": gpu_count, **BASE_RAW_DEPLOYMENT_CONFIG}

    return deployment_config_dict


def get_test_case_id(model_format_name: str, deployment_type: str, protocol_type: str) -> str:
    """
    Generate a test case identifier string based on model format, deployment type, and protocol type.

    Args:
        model_format_name (str): The model format name (e.g., "onnx").
        deployment_type (str): The deployment type (e.g., "raw").
        protocol_type (str): The protocol type (e.g., "rest").

    Returns:
        str: A test case ID in the format: "<model_format>-<deployment_type>-<protocol_type>-deployment".
              Example: "onnx-raw-rest-deployment"
    """
    return f"{model_format_name.strip()}-{deployment_type.strip()}-{protocol_type.strip()}-deployment"


def get_input_query(model_format_config: dict[str, Any], protocol: str) -> dict[str, Any]:
    """
    Get the input query for the given protocol from the model config.

    Looks up a protocol-specific key in the config; loads from JSON file if necessary.

    Args:
        model_format_config: The model format config dictionary.
        protocol: The protocol name (e.g., "REST", "gRPC").

    Returns:
        The input query as a dictionary.

    Raises:
        ValueError: If the input value is neither a dict nor a valid file path.
    """
    query_key = f"{protocol.lower()}_query_or_path"
    input_query_or_path = model_format_config[query_key]

    if isinstance(input_query_or_path, dict):
        return input_query_or_path
    elif os.path.exists(input_query_or_path):
        with open(input_query_or_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Invalid input query or path: {input_query_or_path!r}")
