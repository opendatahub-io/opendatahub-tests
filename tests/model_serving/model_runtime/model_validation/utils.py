import re
from typing import Any

from tests.model_serving.model_runtime.vllm.constant import VLLM_SUPPORTED_QUANTIZATION


def normalize_output(output):
    """
    Recursively normalize model output by removing or masking fields that cause non-deterministic snapshot changes.
    Handles nested dicts and lists.
    """
    if isinstance(output, dict):
        output = output.copy()
        volatile_keys = ["timestamp", "created_at", "updated_at", "id", "unique_id", "request_id", "uuid", "run_id"]
        for key in volatile_keys:
            output.pop(key, None)
        for k, v in output.items():
            output[k] = normalize_output(output=v)
    elif isinstance(output, list):
        output = [normalize_output(output=item) for item in output]
        try:
            output = sorted(output, key=lambda x: str(x))
        except Exception:
            pass
    elif isinstance(output, str):
        output = re.sub(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", "[MASKED_UUID]", output)
    return output


def validate_supported_quantization_schema(q_type: str) -> None:
    if q_type not in VLLM_SUPPORTED_QUANTIZATION:
        raise ValueError(f"Unsupported quantization type: {q_type}")


def validate_inference_output(*args: tuple[Any, ...], response_snapshot: Any) -> None:
    normalized_args = [normalize_output(output=data) for data in args]
    normalized_snapshot = normalize_output(output=response_snapshot)
    for data in normalized_args:
        assert data == normalized_snapshot, f"output mismatch for {data}"


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
