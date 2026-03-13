"""
Utility functions for TRITON model serving tests.

This module provides functions for:
- Managing S3 secrets for model access
- Sending inference requests via REST and gRPC protocols
- Running inference against TRITON deployments
- Validating responses against snapshots
"""

import json
import os
import subprocess
import tempfile
from typing import Any

import portforward
import requests
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_runtime.triton.constant import (
    ACCELERATOR_IDENTIFIER,
    LOCAL_HOST_URL,
    PROTO_FILE_PATH,
    TEMPLATE_MAP,
    TRITON_GRPC_PORT,
    TRITON_REST_PORT,
)
from utilities.constants import KServeDeploymentType, Labels, Protocols, RuntimeTemplates


def send_rest_request(url: str, input_data: dict[str, Any]) -> Any:
    response = requests.post(url=url, json=input_data, verify=False, timeout=180)
    response.raise_for_status()
    return response.json()


def send_grpc_request(url: str, input_data: dict[str, Any], root_dir: str, insecure: bool = False) -> Any:
    """
    Sends a gRPC request using grpcurl.
    Uses inline -d for small payloads and stdin for large payloads.
    """
    grpc_proto_path = os.path.join(root_dir, PROTO_FILE_PATH)
    proto_import_path = os.path.dirname(grpc_proto_path)
    grpc_method = "inference.GRPCInferenceService/ModelInfer"

    input_str = json.dumps(input_data)
    use_stdin = len(input_str.encode("utf-8")) > 8000

    base_args = [
        "grpcurl",
        "-insecure" if insecure else "-plaintext",
        "-import-path",
        proto_import_path,
        "-proto",
        grpc_proto_path,
        url,
        grpc_method,
    ]

    if use_stdin:
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tmpfile:
            tmpfile.write(input_str)
            tmpfile.flush()

            args = base_args.copy()
            args.insert(args.index(url), "-d")
            args.insert(args.index("-d") + 1, "@")

            try:
                with open(tmpfile.name, "r") as f:
                    proc = subprocess.run(
                        args=args,
                        stdin=f,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                return json.loads(proc.stdout)
            except subprocess.CalledProcessError as e:
                return f"gRPC request (stdin) failed:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
            finally:
                os.unlink(tmpfile.name)
    else:
        args = base_args.copy()
        args.insert(args.index(url), "-d")
        args.insert(args.index("-d") + 1, input_str)

        try:
            proc = subprocess.run(
                args=args,
                capture_output=True,
                text=True,
                check=True,
            )
            return json.loads(proc.stdout)
        except subprocess.CalledProcessError as e:
            return f"gRPC request (inline) failed:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"


def run_triton_inference(
    pod_name: str, isvc: InferenceService, input_data: dict[str, Any], model_name: str, protocol: str, root_dir: str
) -> Any:
    deployment_mode = isvc.instance.metadata.annotations.get("serving.kserve.io/deploymentMode")
    rest_endpoint = f"/v2/models/{model_name}/infer"

    if protocol not in (Protocols.REST, Protocols.GRPC):
        return f"Invalid protocol {protocol}"

    is_rest = protocol == Protocols.REST

    if deployment_mode == KServeDeploymentType.RAW_DEPLOYMENT:
        port = TRITON_REST_PORT if is_rest else TRITON_GRPC_PORT
        with portforward.forward(pod_or_service=pod_name, namespace=isvc.namespace, from_port=port, to_port=port):
            host = f"{LOCAL_HOST_URL}:{port}" if is_rest else get_grpc_url(base_url=LOCAL_HOST_URL, port=port)
            return (
                send_rest_request(f"{host}{rest_endpoint}", input_data)
                if is_rest
                else send_grpc_request(host, input_data, root_dir)
            )

    return f"Invalid deployment_mode {deployment_mode}"


def get_grpc_url(base_url: str, port: int) -> str:
    return f"{base_url.replace('https://', '').replace('http://', '')}:{port}"


def validate_inference_request(
    pod_name: str,
    isvc: InferenceService,
    response_snapshot: Any,
    input_query: Any,
    model_name: str,
    protocol: str,
    root_dir: str,
) -> None:
    response = run_triton_inference(
        pod_name=pod_name,
        isvc=isvc,
        input_data=input_query,
        model_name=model_name,
        protocol=protocol,
        root_dir=root_dir,
    )
    assert response == response_snapshot, f"Output mismatch: {response} != {response_snapshot}"


def get_gpu_identifier(accelerator_type: str | None) -> str:
    if accelerator_type is None:
        return Labels.Nvidia.NVIDIA_COM_GPU
    return ACCELERATOR_IDENTIFIER.get(accelerator_type.lower(), Labels.Nvidia.NVIDIA_COM_GPU)


def get_template_name(protocol: str, accelerator_type: str | None) -> str:
    """
    Returns template name based on protocol and accelerator type.
    Falls back to default TRITON_REST if not found.
    If accelerator_type is None, defaults to "nvidia".
    """
    if accelerator_type is None:
        accelerator_type = "nvidia"
    key = f"{protocol.lower()}_{accelerator_type.lower()}"
    return TEMPLATE_MAP.get(key, RuntimeTemplates.TRITON_REST)


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)
