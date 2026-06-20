"""
Utility functions for MLServer model serving tests.

This module provides functions for:
- Sending inference requests via REST protocol
- Running inference against MLServer deployments
- Validating responses against snapshots
- Generating test configuration dictionaries
"""

from typing import Any

import requests
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_runtime.mlserver.constant import (
    BASE_RAW_DEPLOYMENT_CONFIG,
    MODEL_PATH_PREFIX,
    OutputType,
)
from utilities.constants import KServeDeploymentType, Timeout
from utilities.inference_utils import get_exposed_isvc_url


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
    response = requests.post(url=url, json=input_data, verify=verify, timeout=Timeout.TIMEOUT_1MIN)
    response.raise_for_status()
    return response.json()


def run_mlserver_inference(isvc: InferenceService, input_data: dict[str, Any], model_version: str) -> Any:
    """
    Run inference against an MLServer-hosted model using the external route.

    Args:
        isvc (InferenceService): The KServe InferenceService object.
        input_data (dict[str, Any]): The input data payload for inference.
        model_version (str): The version of the model to target, if applicable.

    Returns:
        Any: The inference result from the model.
    """
    model_name = isvc.instance.metadata.name
    version_suffix = f"/versions/{model_version}" if model_version else ""
    rest_endpoint = f"/v2/models/{model_name}{version_suffix}/infer"

    url = get_exposed_isvc_url(isvc=isvc)
    return send_rest_request(url=f"{url}{rest_endpoint}", input_data=input_data, verify=False)


def validate_inference_request(
    isvc: InferenceService,
    response_snapshot: Any,
    input_query: Any,
    model_version: str,
    model_output_type: str,
) -> None:
    """
    Runs an inference request against an MLServer model and validates
    the response using fuzzy validation.
    """
    response = run_mlserver_inference(
        isvc=isvc,
        input_data=input_query,
        model_version=model_version,
    )

    if model_output_type == OutputType.DETERMINISTIC:
        validate_deterministic_snapshot(response=response, response_snapshot=response_snapshot)
    elif model_output_type == OutputType.NON_DETERMINISTIC:
        validate_nondeterministic_snapshot(response=response)


def validate_deterministic_snapshot(response: Any, response_snapshot: Any) -> None:
    """
    Validates a deterministic model inference response using fuzzy validation.

    This function validates the response structure and data presence without comparing
    exact float values, which allows tests to pass on different GPU types (NVIDIA, AMD,
    Gaudi, CPU) that may produce slightly different numerical precision.

    Args:
        response (Any): The actual inference response from the model.
        response_snapshot (Any): The stored snapshot representing the expected output (unused for fuzzy validation).

    Raises:
        AssertionError: If the response structure is invalid or data is empty.
    """
    assert response, "Response is empty"
    assert isinstance(response, dict), f"Response is not a dict: {response}"
    assert response.get("outputs"), "Response missing outputs"
    assert isinstance(response["outputs"], list), "Outputs must be a list"
    assert len(response["outputs"]) > 0, "Outputs list is empty"

    output = response["outputs"][0]
    assert isinstance(output, dict), f"Output must be a dict, got {type(output).__name__}"

    actual_data = output.get("data", [])
    assert actual_data, "Data is empty"
    assert isinstance(actual_data, list), "Data must be a list"
    assert all(isinstance(x, (int, float, list)) for x in actual_data), "Invalid data types in response"


def validate_nondeterministic_snapshot(response: Any) -> None:
    """
    Validates a model inference response containing non-deterministic output.
    Checks that the response contains generated text with expected keywords.
    """
    response_data = ""

    try:
        response_data = response["outputs"][0]["data"][0]

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
    deployment_mode: str = KServeDeploymentType.STANDARD,
) -> dict[str, str]:
    """
    Generate a deployment configuration dictionary based on the model format and deployment mode.

    Args:
        model_format_name (str): The model format name (e.g., "sklearn").
        deployment_mode (str): The deployment mode. Defaults to "Standard".

    Returns:
        dict[str, str]: A dictionary containing the deployment configuration.
    """
    deployment_config_dict = {}

    if deployment_mode == KServeDeploymentType.STANDARD:
        deployment_config_dict = {"name": model_format_name, **BASE_RAW_DEPLOYMENT_CONFIG}

    return deployment_config_dict


def get_test_case_id(
    model_format_name: str,
    deployment_mode: str = KServeDeploymentType.STANDARD,
    modelcar: bool = False,
) -> str:
    """
    Generate a test case identifier string based on model format and deployment mode.

    Args:
        model_format_name (str): The model format name (e.g., "sklearn").
        deployment_mode (str): The deployment mode. Defaults to "Standard".
        modelcar (bool): Whether this is a model car deployment. Defaults to False.

    Returns:
        str: A test case ID. Example: "sklearn-s3-standard" or "sklearn-modelcar-standard"
    """
    storage_type = "modelcar" if modelcar else "s3"
    base_id = f"{model_format_name.strip()}-{storage_type}-{deployment_mode.strip().lower()}"
    return base_id
