"""
Utility functions for MLServer model serving tests.

This module provides functions for:
- Sending inference requests via REST protocol
- Running inference against MLServer deployments
- Validating responses against snapshots
- Generating test configuration dictionaries
"""

from typing import Any

import portforward
import requests
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_runtime.mlserver.constant import (
    BASE_RAW_DEPLOYMENT_CONFIG,
    LOCALHOST_URL,
    MODEL_PATH_PREFIX,
    OutputType,
)
from utilities.constants import KServeDeploymentType, Ports, Protocols


def send_rest_request(url: str, input_data: dict[str, Any], verify: bool = False) -> Any:
    """
    Sends a REST POST request to the specified URL with the given JSON payload.

    Args:
        url (str): The endpoint URL to send the request to.
        input_data (dict[str, Any]): The input payload to send as JSON.
        verify (bool): Whether to verify SSL certificates. Defaults to False.

    Returns:
        Any: The parsed JSON response from the server.

    Raises:
        requests.HTTPError: If the response contains an HTTP error status.
    """
    response = requests.post(url=url, json=input_data, verify=verify, timeout=60)
    response.raise_for_status()
    return response.json()


def run_mlserver_inference(
    pod_name: str, isvc: InferenceService, input_data: dict[str, Any], model_version: str, protocol: str
) -> Any:
    """
    Run inference against an MLServer-hosted model using REST protocol.
    Supports RawDeployment(Standard) modes.

    Args:
        pod_name (str): Name of the pod running the MLServer model.
        isvc (InferenceService): The KServe InferenceService object.
        input_data (dict[str, Any]): The input data payload for inference.
        model_version (str): The version of the model to target, if applicable.
        protocol (str): Protocol to use for inference ('REST').

    Returns:
        Any: The inference result from the model.

    Raises:
        ValueError: If the protocol is not REST or deployment mode is not RAW_DEPLOYMENT.

    Notes:
        - REST calls expect the model to support V2 REST inference APIs.
        - Uses port-forwarding for RawDeployment(Standard) modes.
    """
    deployment_mode = isvc.instance.metadata.annotations.get("serving.kserve.io/deploymentMode")
    model_name = isvc.instance.metadata.name
    version_suffix = f"/versions/{model_version}" if model_version else ""
    rest_endpoint = f"/v2/models/{model_name}{version_suffix}/infer"

    if protocol != Protocols.REST:
        raise ValueError(f"Unsupported protocol: {protocol}. Only REST is supported.")

    supported_modes = (KServeDeploymentType.RAW_DEPLOYMENT, KServeDeploymentType.STANDARD)
    if deployment_mode not in supported_modes:
        raise ValueError(f"Unsupported deployment mode: {deployment_mode}. Supported modes: {supported_modes}")

    port = Ports.REST_PORT
    with portforward.forward(pod_or_service=pod_name, namespace=isvc.namespace, from_port=port, to_port=port):
        host = f"{LOCALHOST_URL}:{port}"
        return send_rest_request(url=f"{host}{rest_endpoint}", input_data=input_data, verify=False)


def validate_inference_request(
    pod_name: str,
    isvc: InferenceService,
    response_snapshot: Any,
    input_query: Any,
    model_version: str,
    model_output_type: str,
    protocol: str,
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
        model_output_type (str): The type of output (deterministic or non_deterministic).
        protocol (str): The protocol to use for inference ('REST').

    Raises:
        AssertionError: If the actual response does not match the snapshot.
    """

    response = run_mlserver_inference(
        pod_name=pod_name,
        isvc=isvc,
        input_data=input_query,
        model_version=model_version,
        protocol=protocol,
    )

    if model_output_type == OutputType.DETERMINISTIC:
        validate_deterministic_snapshot(response=response, response_snapshot=response_snapshot)
    elif model_output_type == OutputType.NON_DETERMINISTIC:
        validate_nondeterministic_snapshot(response=response, protocol=protocol)


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


def validate_nondeterministic_snapshot(response: Any, protocol: str) -> None:
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
        response (Any): The RawDeployment inference response returned by the model server.
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

        assert "generated_text" in response_data, "Keyword 'generated_text' not found in generated text."
        assert "test" in response_data, "Keyword 'test' not found in generated text."

    except Exception as e:
        raise RuntimeError(
            f"Exception in validate_nondeterministic_snapshot: with response_data = {response_data} and exception = {e}"
        ) from e


def get_model_storage_uri_dict(
    model_format_name: str,
    modelcar: bool = False,
    env_variables: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Generate a dictionary containing the storage configuration for a given model format.

    This utility helps build a consistent storage configuration dictionary for both
    S3-based deployments and model car (OCI image) deployments.

    Args:
        model_format_name (str): Name of the model format (e.g., "sklearn").
        modelcar (bool): If True, generate config for model car (OCI image) deployment.
                        If False (default), generate config for S3 storage.
        env_variables (list[dict[str, str]] | None): Optional environment variables for model car deployments.

    Returns:
        dict[str, Any]: For S3 (modelcar=False): {"model-dir": "/mnt/models/sklearn"}
                       For model car (modelcar=True): {"storage-uri": "oci://quay.io/...", "model-format": "sklearn"}
    """
    if modelcar:
        from utilities.constants import ModelCarImage

        attr_name = f"MLSERVER_{model_format_name.upper()}"
        if not hasattr(ModelCarImage, attr_name):
            raise ValueError(
                f"No ModelCarImage constant found for model format '{model_format_name}' (expected {attr_name})"
            )

        storage_uri = getattr(ModelCarImage, attr_name)

        config: dict[str, Any] = {
            "storage-uri": storage_uri,
            "model-format": model_format_name,
        }

        if env_variables:
            config["model_env_variables"] = env_variables

        return config
    else:
        return {"model-dir": f"{MODEL_PATH_PREFIX.rstrip('/')}/{model_format_name.lstrip('/')}"}


def get_model_namespace_dict(
    model_format_name: str,
    modelcar: bool = False,
) -> dict[str, str]:
    """
    Generate a dictionary containing a unique model namespace or name identifier.

    The function constructs a name by concatenating the given model format
    and storage type using hyphens. It is useful for dynamically
    naming model-serving resources, configurations, or deployments.

    Args:
        model_format_name (str): The model format name (e.g., "sklearn").
        modelcar (bool): If True, use "modelcar" suffix defaults.

    Returns:
        dict[str, str]: A dictionary with the key "name" and a concatenated identifier as value.
                        Example: {"name": "sklearn-s3"} or {"name": "sklearn-modelcar"}
    """
    if modelcar:
        name = f"{model_format_name.strip()}-modelcar"
    else:
        name = f"{model_format_name.strip()}-s3"
    return {"name": name}


def get_deployment_config_dict(
    model_format_name: str,
    deployment_mode: str = KServeDeploymentType.RAW_DEPLOYMENT,
) -> dict[str, str]:
    """
    Generate a deployment configuration dictionary based on the model format and deployment mode.

    This function merges a base deployment configuration (RawDeployment) with a given model format
    name to produce a complete configuration dictionary.

    Args:
        model_format_name (str): The model format name (e.g., "sklearn").
        deployment_mode (str): The deployment mode. Defaults to "RawDeployment".

    Returns:
        dict[str, str]: A dictionary containing the deployment configuration.
    """
    deployment_config_dict = {}

    if deployment_mode == KServeDeploymentType.RAW_DEPLOYMENT:
        deployment_config_dict = {"name": model_format_name, **BASE_RAW_DEPLOYMENT_CONFIG}

    return deployment_config_dict


def get_test_case_id(
    model_format_name: str,
    deployment_mode: str = KServeDeploymentType.RAW_DEPLOYMENT,
    modelcar: bool = False,
) -> str:
    """
    Generate a test case identifier string based on model format and deployment mode.

    Args:
        model_format_name (str): The model format name (e.g., "sklearn").
        deployment_mode (str): The deployment mode. Defaults to "RawDeployment".
        modelcar (bool): Whether this is a model car deployment. Defaults to False.

    Returns:
        str: A test case ID in the format: "<model_format>-<storage_type>-<deployment_mode>".
              Example: "sklearn-s3-RawDeployment" or "sklearn-modelcar-RawDeployment"
    """
    storage_type = "modelcar" if modelcar else "s3"
    base_id = f"{model_format_name.strip()}-{storage_type}-{deployment_mode.strip()}"
    return base_id
