"""
Utility functions for TRITON model serving tests.

This module provides functions for:
- Managing S3 secrets for model access
- Sending inference requests via REST and gRPC protocols
- Running inference against TRITON deployments
- Validating responses against snapshots
"""

import os
import json
import subprocess
import tempfile
from contextlib import contextmanager
from typing import Generator, Any

import requests
import portforward

from kubernetes.dynamic import DynamicClient
from ocp_resources.secret import Secret
from ocp_resources.inference_service import InferenceService

from utilities.constants import KServeDeploymentType, Protocols
from tests.model_serving.model_runtime.triton.constant import (
    TRITON_GRPC_REMOTE_PORT,
    LOCAL_HOST_URL,
    PROTO_FILE_PATH,
    TRITON_REST_PORT,
    TRITON_GRPC_PORT,
)


@contextmanager
def kserve_s3_endpoint_secret(
    admin_client: DynamicClient,
    name: str,
    namespace: str,
    aws_access_key: str,
    aws_secret_access_key: str,
    aws_s3_endpoint: str,
    aws_s3_region: str,
) -> Generator[Secret, Any, Any]:
    with Secret(
        client=admin_client,
        name=name,
        namespace=namespace,
        annotations={
            "serving.kserve.io/s3-endpoint": aws_s3_endpoint.replace("https://", ""),
            "serving.kserve.io/s3-region": aws_s3_region,
            "serving.kserve.io/s3-useanoncredential": "false",
            "serving.kserve.io/s3-verifyssl": "0",
            "serving.kserve.io/s3-usehttps": "1",
        },
        string_data={
            "AWS_ACCESS_KEY_ID": aws_access_key,
            "AWS_SECRET_ACCESS_KEY": aws_secret_access_key,
        },
        wait_for_resource=True,
    ) as secret:
        yield secret


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
    pod_name: str, isvc: InferenceService, input_data: dict[str, Any], model_version: str, protocol: str, root_dir: str
) -> Any:
    deployment_mode = isvc.instance.metadata.annotations.get("serving.kserve.io/deploymentMode")
    model_name = isvc.instance.metadata.name
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

    elif deployment_mode == KServeDeploymentType.SERVERLESS:
        base_url = isvc.instance.status.url.rstrip("/")
        if is_rest:
            return send_rest_request(f"{base_url}{rest_endpoint}", input_data)
        else:
            grpc_url = get_grpc_url(base_url=base_url, port=TRITON_GRPC_REMOTE_PORT)
            return send_grpc_request(grpc_url, input_data, root_dir, insecure=True)

    return f"Invalid deployment_mode {deployment_mode}"


def get_grpc_url(base_url: str, port: int) -> str:
    return f"{base_url.replace('https://', '').replace('http://', '')}:{port}"


def validate_inference_request(
    pod_name: str,
    isvc: InferenceService,
    response_snapshot: Any,
    input_query: Any,
    model_version: str,
    protocol: str,
    root_dir: str,
) -> None:
    response = run_triton_inference(
        pod_name=pod_name,
        isvc=isvc,
        input_data=input_query,
        model_version=model_version,
        protocol=protocol,
        root_dir=root_dir,
    )
    assert response == response_snapshot, f"Output mismatch: {response} != {response_snapshot}"


from tests.model_serving.model_runtime.triton.constant import ACCELERATOR_IDENTIFIER, TEMPLATE_MAP
from utilities.constants import Labels, RuntimeTemplates


def get_accelerator_label(accelerator_type: str) -> str:
    """
    Maps accelerator type to Kubernetes GPU identifier. Defaults to NVIDIA.
    """
    return ACCELERATOR_IDENTIFIER.get(accelerator_type.lower(), Labels.Nvidia.NVIDIA_COM_GPU)


def get_template_name(protocol: str, accelerator_type: str) -> str:
    """
    Returns template name based on protocol and accelerator type.
    Falls back to default TRITON_REST if not found.
    """
    key = f"{protocol.lower()}_{accelerator_type.lower()}"
    return TEMPLATE_MAP.get(key, RuntimeTemplates.TRITON_REST)
