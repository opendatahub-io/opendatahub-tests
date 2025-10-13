import re
from typing import Any, Dict, List
from utilities.constants import Protocols
from tests.model_serving.model_runtime.vllm.constant import VLLM_SUPPORTED_QUANTIZATION


def validate_supported_quantization_schema(q_type: str) -> None:
    if q_type not in VLLM_SUPPORTED_QUANTIZATION:
        raise ValueError(f"Unsupported quantization type: {q_type}")


def validate_inference_output(*args: tuple[str, ...] | list[Any], response_snapshot: Any) -> None:
    for data in args:
        assert data == response_snapshot, f"output mismatch for {data}"


def safe_k8s_name(model_name: str, max_length: int = 20) -> str:
    """
    Create a safe Kubernetes name from model_name by truncating to max_length characters
    and ensuring it follows Kubernetes naming conventions.

    Args:
        model_name: The original model name
        max_length: Maximum length for the name (default: 20)

    Returns:
        A valid Kubernetes name truncated to max_length characters
    """
    if not model_name:
        return "default-model"

    # Convert to lowercase and replace invalid characters with hyphens
    safe_name = re.sub(r"[^a-z0-9-]", "-", model_name.lower())

    # Remove consecutive hyphens
    safe_name = re.sub(r"-+", "-", safe_name)

    # Remove leading/trailing hyphens
    safe_name = safe_name.strip("-")

    # Truncate to max_length
    if len(safe_name) > max_length:
        safe_name = safe_name[:max_length]

    # Ensure it doesn't end with a hyphen after truncation
    safe_name = safe_name.rstrip("-")

    # Ensure it's not empty after all processing
    if not safe_name:
        return "model"

    return safe_name


def create_vllm_spyre_serving_runtime(protocol: str, vllm_runtime_image: str) -> dict[str, Any]:
    volumes = []
    volume_mounts = []
    if protocol == Protocols.GRPC:
        volumes.append({"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"}})
        volume_mounts.append({"name": "shm", "mountPath": "/dev/shm"})

    port_config = {
        "name": "h2c" if protocol == Protocols.GRPC else "http1",
        "containerPort": 9000 if protocol == Protocols.GRPC else 8000,
        "protocol": "TCP",
    }

    container_args = ["--served-model-name={{.Name}}", "--model=/mnt/models", "--port=8000"]

    container_command = [
        "- /bin/bash",
        "- -c",
        "- |",
        "source /etc/profile.d/ibm-aiu-setup.sh && ",
        'exec python3 -m vllm.entrypoints.openai.api_server "$@"',
    ]

    env_variables = [
        {"name": "HF_HOME", "value": "/tmp/hf_home"},
        {"name": "FLEX_COMPUTE", "value": "SENTIENT"},
        {"name": "FLEX_DEVICE", "value": "PF"},
        {"name": "TOKENIZERS_PARALLELISM", "value": "false"},
        {"name": "DTLOG_LEVEL", "value": "error"},
        {"name": "TORCH_SENDNN_LOG", "value": "CRITICAL"},
        {"name": "VLLM_SPYRE_WARMUP_BATCH_SIZES", "value": "4"},
        {"name": "VLLM_SPYRE_WARMUP_PROMPT_LENS", "value": "1024"},
        {"name": "VLLM_SPYRE_WARMUP_NEW_TOKENS", "value": "256"},
    ]

    kserve_container: List[Dict[str, Any]] = [
        {
            "name": "vllm",
            "image": vllm_runtime_image,
            "ports": [port_config],
            "command": container_command,
            "args": container_args,
            "volumeMounts": volume_mounts,
            "resources": {
                "requests": {
                    "ibm.com/spyre_pf": "1",
                },
                "limits": {
                    "ibm.com/spyre_pf": "1",
                },
            },
            "env": env_variables,
        }
    ]

    supported_model_formats = List[Dict[str, Any]] = [
        {
            "name": "vLLM",
            "autoSelect": True,
        }
    ]

    return {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "ServingRuntime",
        "metadata": {
            "name": "vllm-spyre-runtime",
            "annotations": {
                "openshift.io/display-name": "vLLM IBM Spyre ServingRuntime for KServe",
                "opendatahub.io/recommended-accelerators": '["ibm.com/spyre_pf"]',
            },
        },
        "spec": {
            "annotations": {
                "prometheus.io/port": "8080",
                "prometheus.io/path": "/metrics",
            },
            "containers": kserve_container,
            "volumes": volumes,
            "supportedModelFormats": supported_model_formats,
        },
    }
