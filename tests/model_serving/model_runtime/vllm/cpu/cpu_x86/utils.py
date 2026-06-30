from typing import Any

import requests
import structlog
from ocp_resources.inference_service import InferenceService
from tenacity import retry, stop_after_attempt, wait_exponential

from utilities.certificates_utils import get_ca_bundle
from utilities.inference_utils import get_exposed_isvc_url
from utilities.plugins.constant import OpenAIEnpoints, RestHeader

LOGGER = structlog.get_logger(name=__name__)

DEFAULT_REQUEST_TIMEOUT: int = 120


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=6))
def send_completions_request(
    isvc: InferenceService,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Send a POST request to /v1/completions matching the OpenAI completions API."""
    url = f"{get_exposed_isvc_url(isvc=isvc)}{OpenAIEnpoints.COMPLETIONS}"
    payload = {
        "model": isvc.instance.metadata.name,
        "prompt": prompt,
        "max_tokens": max_tokens,
    }
    LOGGER.info("Sending completions request", url=url, payload=payload)
    ca_bundle = get_ca_bundle(client=isvc.client)
    response = requests.post(
        url=url,
        headers=RestHeader.HEADERS,
        json=payload,
        verify=ca_bundle or True,
        timeout=DEFAULT_REQUEST_TIMEOUT,
    )
    LOGGER.info("Completions response", status_code=response.status_code, body_length=len(response.text))
    response.raise_for_status()
    return response.json()


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=6))
def send_chat_completions_request(
    isvc: InferenceService,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> dict[str, Any]:
    """Send a POST request to /v1/chat/completions matching the OpenAI chat API."""
    url = f"{get_exposed_isvc_url(isvc=isvc)}{OpenAIEnpoints.CHAT_COMPLETIONS}"
    payload = {
        "model": isvc.instance.metadata.name,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    LOGGER.info("Sending chat completions request", url=url, payload=payload)
    ca_bundle = get_ca_bundle(client=isvc.client)
    response = requests.post(
        url=url,
        headers=RestHeader.HEADERS,
        json=payload,
        verify=ca_bundle or True,
        timeout=DEFAULT_REQUEST_TIMEOUT,
    )
    LOGGER.info("Chat completions response", status_code=response.status_code, body_length=len(response.text))
    response.raise_for_status()
    return response.json()


def validate_cpu_x86_completions_request(
    isvc: InferenceService,
    prompt: str,
    max_tokens: int,
) -> None:
    """Validate that the InferenceService returns a non-empty /v1/completions response."""
    body = send_completions_request(isvc=isvc, prompt=prompt, max_tokens=max_tokens)
    completion_text = body["choices"][0]["text"]
    assert completion_text.strip(), f"Expected non-empty completion text, got: {body!r}"


def validate_cpu_x86_chat_completions_request(
    isvc: InferenceService,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> None:
    """Validate that the InferenceService returns a non-empty /v1/chat/completions response."""
    body = send_chat_completions_request(isvc=isvc, messages=messages, max_tokens=max_tokens)
    completion_text = body["choices"][0]["message"]["content"]
    assert completion_text.strip(), f"Expected non-empty chat completion text, got: {body!r}"


def validate_cpu_x86_inference_request(
    isvc: InferenceService,
    inference_request: dict[str, Any],
) -> None:
    """Validate inference using /v1/completions or /v1/chat/completions based on the request shape."""
    max_tokens = int(inference_request["max_tokens"])
    if messages := inference_request.get("messages"):
        validate_cpu_x86_chat_completions_request(isvc=isvc, messages=messages, max_tokens=max_tokens)
        return

    validate_cpu_x86_completions_request(
        isvc=isvc,
        prompt=inference_request["prompt"],
        max_tokens=max_tokens,
    )
