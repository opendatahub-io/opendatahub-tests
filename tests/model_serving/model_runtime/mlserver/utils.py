"""
Utility functions for MLServer model serving tests.

This module provides functions for:
- Managing S3 secrets for model access
- Sending inference requests via REST protocol
- Running inference against MLServer deployments
- Validating responses against snapshots
"""

from typing import Any, Dict

import portforward
import requests
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_runtime.mlserver.constant import (
    LOCAL_HOST_URL,
    MLSERVER_REST_PORT,
    DETERMINISTIC_OUTPUT,
    NON_DETERMINISTIC_OUTPUT,
    MODEL_PATH_PREFIX,
    RAW_DEPLOYMENT_TYPE,
    BASE_RAW_DEPLOYMENT_CONFIG,
    MLSERVER_RUNTIME_LABELS,
    MLSERVER_RUNTIME_ANNOTATIONS,
    MLSERVER_SUPPORTED_MODEL_FORMATS,
    MLSERVER_IMAGE,
    MLSERVER_CONTAINER_ENV,
    MLSERVER_CONTAINER_SECURITY_CONTEXT,
    MLSERVER_PORTS_MAP,
    TEMPLATE_NAME_MAP,
    RUNTIME_NAME_MAP,
)
from utilities.constants import KServeDeploymentType, Protocols


def send_rest_request(url: str, input_data: dict[str, Any], verify: bool = False) -> Any:
    """
    Sends a REST POST request to the specified URL with the given JSON payload.

    Args:
        url (str): The endpoint URL to send the request to.
        input_data (dict[str, Any]): The input payload to send as JSON.

    Returns:
        Any: The parsed JSON response from the server.

    Raises:
        requests.HTTPError: If the response contains an HTTP error status.
    """
    response = requests.post(url=url, json=input_data, verify=verify, timeout=60)
    response.raise_for_status()
    return response.json()


def run_mlserver_inference(
    pod_name: str, isvc: InferenceService, input_data: dict[str, Any], model_version: str, protocol: str, root_dir: str
) -> Any:
    """
    Run inference against an MLServer-hosted model using REST protocol.
    Supports RAW KServe deployment mode.

    Args:
        pod_name (str): Name of the pod running the MLServer model (used for RAW deployment).
        isvc (InferenceService): The KServe InferenceService object.
        input_data (dict[str, Any]): The input data payload for inference.
        model_version (str): The version of the model to target, if applicable.
        protocol (str): Protocol to use for inference ('REST').
        root_dir (str): Root directory (unused, kept for compatibility).

    Returns:
        Any: The inference result from the model, or an error message string.

    Notes:
        - REST calls expect the model to support V2 REST inference APIs.
        - RAW deployments use port-forwarding.
    """
    deployment_mode = isvc.instance.metadata.annotations.get("serving.kserve.io/deploymentMode")
    model_name = isvc.instance.metadata.name
    version_suffix = f"/versions/{model_version}" if model_version else ""
    rest_endpoint = f"/v2/models/{model_name}{version_suffix}/infer"

    if protocol != Protocols.REST:
        return f"Invalid protocol {protocol}"

    if deployment_mode == KServeDeploymentType.RAW_DEPLOYMENT:
        port = MLSERVER_REST_PORT
        with portforward.forward(pod_or_service=pod_name, namespace=isvc.namespace, from_port=port, to_port=port):
            host = f"{LOCAL_HOST_URL}:{port}"
            return send_rest_request(url=f"{host}{rest_endpoint}", input_data=input_data, verify=False)

    return f"Invalid deployment_mode {deployment_mode}"


def validate_inference_request(
    pod_name: str,
    isvc: InferenceService,
    response_snapshot: Any,
    input_query: Any,
    model_version: str,
    model_framework: str,
    model_output_type: str,
    protocol: str,
    root_dir: str,
) -> None:
    """
    Runs an inference request against an MLServer model and validates
    that the response matches the expected snapshot.

    Args:
        pod_name (str): The pod name where the model is running.
        isvc (InferenceService): The KServe InferenceService instance.
        response_snapshot (Any): The expected inference output to compare against.
        input_query (Any): The input data to send to the model.
        model_version (str): The version of the model to target.
        model_framework (str): The model framework identifier.
        model_output_type (str): The type of output (deterministic or non_deterministic).
        protocol (str): The protocol to use for inference ('REST').
        root_dir (str): The root directory (unused, kept for compatibility).

    Raises:
        AssertionError: If the actual response does not match the snapshot.
    """

    response = run_mlserver_inference(
        pod_name=pod_name,
        isvc=isvc,
        input_data=input_query,
        model_version=model_version,
        protocol=protocol,
        root_dir=root_dir,
    )

    if model_output_type == DETERMINISTIC_OUTPUT:
        validate_deterministic_snapshot(response=response, response_snapshot=response_snapshot)
    elif model_output_type == NON_DETERMINISTIC_OUTPUT:
        validate_nondeterministic_snapshot(response=response, model_framework=model_framework, protocol=protocol)


def validate_deterministic_snapshot(response: Any, response_snapshot: Any) -> None:
    """
    Validates a deterministic model inference response against a stored snapshot.

    This function asserts that the actual model response exactly matches the expected
    snapshot. It is intended for use in scenarios where the model output is expected
    to be consistent across runs, such as with deterministic decoding (e.g., greedy search)
    or fixed seed configurations.

    Args:
        response (Any): The actual inference response from the model.
        response_snapshot (Any): The stored snapshot representing the expected output.

    Raises:
        AssertionError: If the actual response does not exactly match the expected snapshot.
    """
    assert response == response_snapshot, f"Output mismatch: {response} != {response_snapshot}"


def validate_nondeterministic_snapshot(response: Any, model_framework: str, protocol: str) -> None:
    """
    Validates a model inference response containing non-deterministic output.

    This function handles responses returned over REST protocol and extracts generated
    output from a standard prediction response structure. It expects the output to be
    a plain JSON string with the actual generated text stored under a "generated_text" key.

    The function asserts that the generated output contains the keyword "test" as a
    basic form of content validation. This is useful for verifying that the model is
    producing reasonable and expected outputs in snapshot or integration tests, especially
    when exact output matching is not feasible due to variability.

    Args:
        response (Any): The raw inference response returned by the model server.
        model_framework (str): The identifier for the ML framework used.
        protocol (str): The communication protocol used to interact with the model server (e.g., 'rest').

    Raises:
        RuntimeError: If response extraction or keyword validation fails.
    """
    response_data = ""

    try:
        if protocol == Protocols.REST:
            response_data = response["outputs"][0]["data"][0]
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")

        # Validate non-deterministic output (currently no models use this, but kept for future use)
        assert "generated_text" in response_data, "Keyword 'generated_text' not found in generated text."
        assert "test" in response_data, "Keyword 'test' not found in generated text."

    except Exception as e:
        raise RuntimeError(
            f"Exception in validate_nondeterministic_snapshot: with response_data = {response_data} and exception = {e}"
        ) from e


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
        model_format_name (str): The model format name (e.g., "sklearn").
        deployment_type (str): The type of deployment (e.g., "raw").
        protocol_type (str): The communication protocol (e.g., "rest").

    Returns:
        dict[str, str]: A dictionary with the key "name" and a concatenated identifier as value.
                        Example: {"name": "sklearn-raw-rest"}
    """
    name = f"{model_format_name.strip()}-{deployment_type.strip()}-{protocol_type.strip()}"
    return {"name": name}


def get_deployment_config_dict(model_format_name: str, deployment_type: str) -> dict[str, str]:
    """
    Generate a deployment configuration dictionary based on the model format and deployment type.

    This function merges a base deployment configuration (raw) with a given model format
    name to produce a complete configuration dictionary.

    Args:
        model_format_name (str): The model format name (e.g., "sklearn").
        deployment_type (str): The deployment type (e.g., "raw").

    Returns:
        dict[str, str]: A dictionary containing the deployment configuration.
    """
    deployment_config_dict = {}

    if deployment_type == RAW_DEPLOYMENT_TYPE:
        deployment_config_dict = {"name": model_format_name, **BASE_RAW_DEPLOYMENT_CONFIG}

    return deployment_config_dict


def get_test_case_id(model_format_name: str, deployment_type: str, protocol_type: str) -> str:
    """
    Generate a test case identifier string based on model format, deployment type, and protocol type.

    Args:
        model_format_name (str): The model format name (e.g., "sklearn").
        deployment_type (str): The deployment type (e.g., "raw").
        protocol_type (str): The protocol type (e.g., "rest").

    Returns:
        str: A test case ID in the format: "<model_format>-<deployment_type>-<protocol_type>-deployment".
              Example: "sklearn-raw-rest-deployment"
    """
    return f"{model_format_name.strip()}-{deployment_type.strip()}-{protocol_type.strip()}-deployment"


def mlserver_runtime_template_dict(protocol: str) -> Dict[str, Any]:
    """
    Build MLSERVER ServingRuntime template dict for REST protocol.
    Args:
        protocol: "rest"
    Returns:
        Dict representing ServingRuntime template
    """
    if protocol != "rest":
        raise ValueError("protocol must be 'rest'")

    # Port differs based on protocol
    ports_map = MLSERVER_PORTS_MAP.get(protocol, MLSERVER_PORTS_MAP[Protocols.REST])

    return {
        "metadata": {"name": TEMPLATE_NAME_MAP.get(protocol)},
        "objects": [
            {
                "apiVersion": "serving.kserve.io/v1alpha1",
                "kind": "ServingRuntime",
                "metadata": {
                    "name": RUNTIME_NAME_MAP.get(protocol),
                    "labels": MLSERVER_RUNTIME_LABELS,
                },
                "spec": {
                    "annotations": MLSERVER_RUNTIME_ANNOTATIONS,
                    "multiModel": False,
                    "protocolVersions": ["v2"],
                    "supportedModelFormats": MLSERVER_SUPPORTED_MODEL_FORMATS,
                    "containers": [
                        {
                            "name": "kserve-container",
                            "image": MLSERVER_IMAGE,
                            "env": MLSERVER_CONTAINER_ENV,
                            "resources": {
                                "requests": {"cpu": "1", "memory": "2Gi"},
                                "limits": {"cpu": "1", "memory": "2Gi"},
                            },
                            "ports": ports_map,
                            "securityContext": MLSERVER_CONTAINER_SECURITY_CONTEXT,
                        }
                    ],
                },
            }
        ],
    }
