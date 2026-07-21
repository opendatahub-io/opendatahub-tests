import time
from typing import Any

import portforward
import requests
import structlog
from ocp_resources.inference_service import InferenceService
from tenacity import retry, stop_after_attempt, wait_exponential

from utilities.infra import get_pods_by_isvc_label
from utilities.plugins.constant import OpenAIEnpoints

LOGGER = structlog.get_logger(name=__name__)

# Port vLLM listens on inside the predictor pod.
_VLLM_POD_PORT: int = 8080

# Default HTTP client timeout for all IBM Power/Z inference requests (seconds).
# Any model slower on CPU can override this via inference_request["request_timeout"].
DEFAULT_REQUEST_TIMEOUT: int = 300


def _send_via_portforward(
    isvc: InferenceService,
    endpoint: str,
    payload: dict[str, Any],
    request_timeout: int,
) -> dict[str, Any]:
    """POST to the vLLM pod directly via kubectl port-forward, bypassing HAProxy.

    The ODH/KServe controller enforces ``haproxy.router.openshift.io/timeout=30s``
    on the external Route and reconciles it back on every change.  Port-forwarding
    to the predictor pod bypasses HAProxy entirely and is not subject to any router
    timeout, which prevents 504 errors on slow CPU models.

    Args:
        isvc: The InferenceService whose predictor pod is targeted.
        endpoint: Path after the host, e.g. ``"/v1/chat/completions"``.
        payload: JSON-serialisable request body dict.
        request_timeout: HTTP client timeout in seconds.

    Returns:
        Parsed JSON response body.

    Raises:
        RuntimeError: If no predictor pod is found.
        requests.HTTPError: On non-2xx responses.
    """
    pods = get_pods_by_isvc_label(client=isvc.client, isvc=isvc)
    if not pods:
        raise RuntimeError(f"No predictor pod found for InferenceService {isvc.name}")

    pod = pods[0]
    url = f"http://localhost:{_VLLM_POD_PORT}{endpoint}"

    LOGGER.info(
        "Sending request via port-forward (bypassing HAProxy)",
        pod=pod.name,
        url=url,
        model=payload.get("model"),
        request_timeout=request_timeout,
    )

    start_time = time.time()

    with portforward.forward(
        pod_or_service=pod.name,
        namespace=pod.namespace,
        from_port=_VLLM_POD_PORT,
        to_port=_VLLM_POD_PORT,
    ):
        response = requests.post(
            url=url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=request_timeout,
        )

    elapsed_seconds = round(time.time() - start_time, 2)

    LOGGER.info(
        "Port-forward response",
        status_code=response.status_code,
        elapsed_seconds=elapsed_seconds,
    )
    if not response.ok:
        LOGGER.error(
            "Port-forward error body",
            status_code=response.status_code,
            body=response.text,
        )
    response.raise_for_status()
    return response.json()


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=6))
def send_chat_completions_request(
    isvc: InferenceService,
    messages: list[dict[str, str]],
    max_tokens: int,
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
) -> dict[str, Any]:
    """Send a POST request to /v1/chat/completions.

    Uses port-forward to the predictor pod directly, bypassing the external
    OpenShift Route and its HAProxy timeout to prevent 504 errors on slow
    CPU models.

    Args:
        isvc: The InferenceService to send the request to.
        messages: OpenAI-format message dicts.
        max_tokens: Maximum tokens the model may generate.
        request_timeout: HTTP client timeout in seconds. Defaults to
            :data:`DEFAULT_REQUEST_TIMEOUT`. Pass a higher value for large
            models (e.g. Phi-4 14B) that take longer to respond on CPU.
    """
    payload: dict[str, Any] = {
        "model": isvc.instance.metadata.name,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    return _send_via_portforward(
        isvc=isvc,
        endpoint=OpenAIEnpoints.CHAT_COMPLETIONS,
        payload=payload,
        request_timeout=request_timeout,
    )


def validate_ibm_power_z_chat_completions_request(
    isvc: InferenceService,
    inference_request: dict[str, Any],
) -> None:
    """Validate that the InferenceService returns a non-empty /v1/chat/completions response.

    Args:
        isvc: The InferenceService under test.
        inference_request: Dict with keys ``messages``, ``max_tokens``, and
            optionally ``request_timeout`` (int, seconds) to override the
            default HTTP client timeout for slow models such as Phi-4.
    """
    max_tokens = int(inference_request["max_tokens"])
    messages: list[dict[str, str]] = inference_request["messages"]
    request_timeout: int = int(inference_request.get("request_timeout", DEFAULT_REQUEST_TIMEOUT))
    body = send_chat_completions_request(
        isvc=isvc,
        messages=messages,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
    )
    completion_text = body["choices"][0]["message"]["content"]
    assert completion_text.strip(), f"Expected non-empty chat completion text, got: {body!r}"
