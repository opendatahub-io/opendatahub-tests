"""Malformed inference requests to a running vLLM ISVC."""

from typing import Any

import pytest
import requests
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_runtime.vllm.constant import BASE_RAW_DEPLOYMENT_CONFIG
from utilities.inference_utils import get_exposed_isvc_url

pytestmark = pytest.mark.usefixtures("valid_aws_config")

MALFORMED_CHAT_QUERY: dict[str, Any] = {
    "model": "PLACEHOLDER",
    "messages": "this-should-be-a-list-not-a-string",
    "max_tokens": 10,
}

MISSING_MODEL_QUERY: dict[str, Any] = {
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10,
}

EMPTY_BODY_QUERY: dict[str, Any] = {}


@pytest.mark.tier3
@pytest.mark.negative
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, serving_runtime, vllm_inference_service",
    [
        pytest.param(
            {"name": "vllm-neg-proxy"},
            {"model-dir": "vllm/granite-7b-lab"},
            {
                "deployment_mode": BASE_RAW_DEPLOYMENT_CONFIG["deployment_mode"],
            },
            {
                "name": "vllm-error-proxy",
                **BASE_RAW_DEPLOYMENT_CONFIG,
                "gpu_count": 1,
            },
            id="vllm-error-proxying-negative",
        ),
    ],
    indirect=True,
)
class TestErrorProxying:
    """Validate vLLM error responses for malformed and invalid requests."""

    def test_malformed_chat_request(
        self,
        vllm_inference_service: InferenceService,
    ) -> None:
        """Verify vLLM returns 400/422 for a malformed chat completion request."""
        url = get_exposed_isvc_url(isvc=vllm_inference_service)
        query = {**MALFORMED_CHAT_QUERY, "model": vllm_inference_service.name}
        response = requests.post(f"{url}/v1/chat/completions", json=query, verify=False, timeout=30)
        assert response.status_code in (400, 422), (
            f"Expected 400 or 422 for malformed request, got {response.status_code}: {response.text}"
        )

    def test_empty_body_request(
        self,
        vllm_inference_service: InferenceService,
    ) -> None:
        """Verify vLLM returns 400/422 for an empty JSON request body."""
        url = get_exposed_isvc_url(isvc=vllm_inference_service)
        response = requests.post(f"{url}/v1/chat/completions", json=EMPTY_BODY_QUERY, verify=False, timeout=30)
        assert response.status_code in (400, 422), (
            f"Expected 400 or 422 for empty body, got {response.status_code}: {response.text}"
        )

    def test_nonexistent_endpoint(
        self,
        vllm_inference_service: InferenceService,
    ) -> None:
        """Verify vLLM returns 404/405 for a non-existent endpoint."""
        url = get_exposed_isvc_url(isvc=vllm_inference_service)
        response = requests.post(f"{url}/v1/nonexistent", json={"test": "data"}, verify=False, timeout=30)
        assert response.status_code in (404, 405), (
            f"Expected 404 or 405 for non-existent endpoint, got {response.status_code}: {response.text}"
        )
