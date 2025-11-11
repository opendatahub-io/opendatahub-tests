import re
from typing import Any
from tests.model_serving.model_runtime.vllm.constant import VLLM_SUPPORTED_QUANTIZATION


def extract_content_field(output: Any) -> str:
    """
    Extract the 'content' field from a typical VLLM chat response.
    Returns an empty string if not found.
    """
    try:
        return output["choices"][0]["message"].get("content", "").strip()
    except (KeyError, IndexError, TypeError):
        return ""


def validate_supported_quantization_schema(q_type: str) -> None:
    """
    Validate that the quantization type is supported by VLLM.
    """
    if q_type not in VLLM_SUPPORTED_QUANTIZATION:
        raise ValueError(f"Unsupported quantization type: {q_type}")


def validate_inference_output(response_output: Any, expected_keywords: list[str]) -> None:
    """
    Validate inference response output using regex-based keyword checks.

    - Extracts 'content' field from model output.
    - Fails if content is empty.
    - Passes if any of the expected keywords/phrases are found in the content (case-insensitive).
    """
    content = extract_content_field(output=response_output)
    assert content, "Inference output is empty or missing 'content' field."
    found_keywords = [kw for kw in expected_keywords if re.search(rf"\b{re.escape(kw)}\b", content, re.IGNORECASE)]
    assert found_keywords, f"Expected one of {expected_keywords} in response content, but got: {content[:900]}"

    print(f"Output validation passed. Found keywords: {found_keywords}")


def safe_k8s_name(model_name: str, max_length: int = 20) -> str:
    """
    Generate a Kubernetes-safe model name.
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
