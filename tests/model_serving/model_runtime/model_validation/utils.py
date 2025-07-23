import re
from typing import Any

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
