"""
Utility functions for MLServer model serving tests.

This module provides functions for:
- Managing S3 secrets for model access
- Sending inference requests via REST and gRPC protocols
- Running inference against MLServer deployments
- Validating responses against snapshots
"""

import base64
import json
import os
import subprocess
from typing import Any

import portforward
import requests
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_runtime.mlserver.constant import (
    MLSERVER_GRPC_REMOTE_PORT,
    LOCAL_HOST_URL,
    PROTO_FILE_PATH,
    MLSERVER_REST_PORT,
    MLSERVER_GRPC_PORT,
    DETERMINISTIC_OUTPUT,
    NON_DETERMINISTIC_OUTPUT,
    HUGGING_FACE_FRAMEWORK,
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


def send_grpc_request(url: str, input_data: dict[str, Any], root_dir: str, insecure: bool = False) -> Any:
    """
    Sends a gRPC request to the specified URL using grpcurl with the given input data.

    Args:
        url (str): The gRPC server endpoint (host:port).
        input_data (dict[str, Any]): The input payload to send, as a dictionary.
        root_dir (str): Root directory where the .proto file is located.
        insecure (bool, optional): Whether to disable TLS verification.
                                   Defaults to False (uses plaintext).

    Returns:
        Any: The parsed JSON response if successful, or an error message string if the request fails.
    """
    grpc_proto_path = os.path.join(root_dir, PROTO_FILE_PATH)
    proto_import_path = os.path.dirname(grpc_proto_path)
    input_str = json.dumps(input_data)
    grpc_method = "inference.GRPCInferenceService/ModelInfer"
    tls_flag = "-insecure" if insecure else "-plaintext"

    args = [
        "grpcurl",
        tls_flag,
        "-import-path",
        proto_import_path,
        "-proto",
        grpc_proto_path,
        "-d",
        input_str,
        url,
        grpc_method,
    ]

    try:
        result = subprocess.run(args=args, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        return f"gRPC request failed: {e.stderr or e.stdout}"


def run_mlserver_inference(
    pod_name: str, isvc: InferenceService, input_data: dict[str, Any], model_version: str, protocol: str, root_dir: str
) -> Any:
    """
    Run inference against an MLServer-hosted model using either REST or gRPC protocol.
    Supports both RAW and SERVERLESS KServe deployment modes.

    Args:
        pod_name (str): Name of the pod running the MLServer model (used for RAW deployment).
        isvc (InferenceService): The KServe InferenceService object.
        input_data (dict[str, Any]): The input data payload for inference.
        model_version (str): The version of the model to target, if applicable.
        protocol (str): Protocol to use for inference ('REST' or 'GRPC').
        root_dir (str): Root directory containing the .proto file for gRPC requests.

    Returns:
        Any: The inference result from the model, or an error message string.

    Notes:
        - REST calls expect the model to support V2 REST inference APIs.
        - gRPC calls use `grpcurl` and require the appropriate `.proto` files.
        - RAW deployments use port-forwarding; SERVERLESS assumes accessible endpoints.
    """
    deployment_mode = isvc.instance.metadata.annotations.get("serving.kserve.io/deploymentMode")
    model_name = isvc.instance.metadata.name
    version_suffix = f"/versions/{model_version}" if model_version else ""
    rest_endpoint = f"/v2/models/{model_name}{version_suffix}/infer"

    if protocol not in (Protocols.REST, Protocols.GRPC):
        return f"Invalid protocol {protocol}"

    is_rest = protocol == Protocols.REST

    if deployment_mode == KServeDeploymentType.RAW_DEPLOYMENT:
        port = MLSERVER_REST_PORT if is_rest else MLSERVER_GRPC_PORT
        with portforward.forward(pod_or_service=pod_name, namespace=isvc.namespace, from_port=port, to_port=port):
            host = f"{LOCAL_HOST_URL}:{port}" if is_rest else get_grpc_url(base_url=LOCAL_HOST_URL, port=port)
            return (
                send_rest_request(url=f"{host}{rest_endpoint}", input_data=input_data, verify=False)
                if is_rest
                else send_grpc_request(url=host, input_data=input_data, root_dir=root_dir, insecure=False)
            )

    elif deployment_mode == KServeDeploymentType.SERVERLESS:
        base_url = isvc.instance.status.url.rstrip("/")
        if is_rest:
            return send_rest_request(url=f"{base_url}{rest_endpoint}", input_data=input_data, verify=False)
        else:
            grpc_url = get_grpc_url(base_url=base_url, port=MLSERVER_GRPC_REMOTE_PORT)
            return send_grpc_request(url=grpc_url, input_data=input_data, root_dir=root_dir, insecure=True)

    return f"Invalid deployment_mode {deployment_mode}"


def get_grpc_url(base_url: str, port: int) -> str:
    """
    Constructs a gRPC target URL by stripping the HTTP/HTTPS scheme and appending the port.

    Args:
        base_url (str): The base URL, potentially including 'http://' or 'https://'.
        port (int): The port number to append.

    Returns:
        str: A gRPC-compatible URL in the format 'host:port'.
    """
    return f"{base_url.replace('https://', '').replace('http://', '')}:{port}"


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
        protocol (str): The protocol to use for inference ('REST' or 'GRPC').
        root_dir (str): The root directory containing protobuf files for gRPC.

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

    This function handles responses returned over different protocols (REST or gRPC)
    and extracts generated output from a standard prediction response structure. It
    expects the output to be either a plain JSON string (in REST) or a base64-encoded
    JSON string (in gRPC), with the actual generated text stored under a "generated_text" key.

    The function asserts that the generated output contains the keyword "test" as a
    basic form of content validation. This is useful for verifying that the model is
    producing reasonable and expected outputs in snapshot or integration tests, especially
    when exact output matching is not feasible due to variability.

    Args:
        response (Any): The raw inference response returned by the model server.
        model_framework (str): The identifier for the ML framework used.
        protocol (str): The communication protocol used to interact with the model server (e.g., 'rest', 'grpc').

    Raises:
        RuntimeError: If response extraction or keyword validation fails.
    """
    response_data = ""

    try:
        if protocol == Protocols.REST:
            response_data = response["outputs"][0]["data"][0]

        elif protocol == Protocols.GRPC:
            b64_encoded = response["outputs"][0]["contents"]["bytesContents"][0]
            decoded_bytes = base64.b64decode(b64_encoded)
            response_data = decoded_bytes.decode("utf-8")

        if model_framework == HUGGING_FACE_FRAMEWORK:
            assert "generated_text" in response_data, "Keyword 'generated_text' not found in generated text."
            assert "test" in response_data, "Keyword 'test' not found in generated text."

    except Exception as e:
        raise RuntimeError(
            f"Exception in validate_nondeterministic_snapshot: with response_data = {response_data} and exception = {e}"
        ) from e
