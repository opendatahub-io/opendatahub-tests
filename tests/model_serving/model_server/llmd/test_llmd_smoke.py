import pytest

from tests.model_serving.model_server.llmd.llmd_configs import TinyLlamaOciConfig, TinyLlamaS3ConnectionSmokeConfig
from tests.model_serving.model_server.llmd.utils import (
    ns_from_file,
    parse_completion_text,
    send_chat_completions,
)
from utilities.resources.llm_inference_service import LLMInferenceService

pytestmark = [pytest.mark.smoke]

NAMESPACE = ns_from_file(file=__file__)


class TestLLMDSmoke:
    """Smoke tests: deploy TinyLlama on CPU via OCI and verify chat completions, and separately
    assert S3 ConnectionsAPI injection only (no readiness wait, no inference)."""

    @pytest.mark.parametrize(
        "unprivileged_model_namespace, llmisvc",
        [pytest.param({"name": NAMESPACE}, TinyLlamaOciConfig, id="smoke")],
        indirect=True,
    )
    def test_llmd_smoke(self, llmisvc: LLMInferenceService):
        """Test steps:

        1. Send a chat completion request to /v1/chat/completions.
        2. Assert the response status is 200.
        3. Assert the completion text contains the expected answer.
        """
        prompt = "What is the capital of Italy?"
        expected = "rome"

        status, body = send_chat_completions(llmisvc=llmisvc, prompt=prompt)
        assert status == 200, f"Expected 200, got {status}: {body}"
        completion = parse_completion_text(response_body=body)
        assert expected in completion.lower(), f"Expected '{expected}' in response, got: {completion}"

    @pytest.mark.parametrize(
        "unprivileged_model_namespace, llmisvc",
        [pytest.param({"name": NAMESPACE}, TinyLlamaS3ConnectionSmokeConfig, id="s3-connection-smoke")],
        indirect=True,
    )
    def test_llmd_smoke_s3_connection_injects(self, llmisvc: LLMInferenceService):
        """Test steps:

        1. Assert the ConnectionsAPI webhook injected the expected storage/SA fields — no
           readiness wait, no inference.
        """
        llmisvc.connections_config.verify_injection(llmisvc=llmisvc)
