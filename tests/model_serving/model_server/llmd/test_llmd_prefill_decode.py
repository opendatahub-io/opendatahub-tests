import pytest
from ocp_resources.llm_inference_service import LLMInferenceService

from tests.model_serving.model_server.llmd.llmd_configs import PrefillDecodeConfig
from tests.model_serving.model_server.llmd.utils import (
    ns_from_file,
    parse_completion_text,
    send_chat_completions,
)

pytestmark = [pytest.mark.tier2, pytest.mark.gpu]

NAMESPACE = ns_from_file(file=__file__)


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmisvc",
    [({"name": NAMESPACE}, PrefillDecodeConfig)],
    indirect=True,
)
@pytest.mark.usefixtures("valid_aws_config", "skip_if_no_gpu_available", "skip_if_disconnected")
class TestLlmdPrefillDecode:
    """Deploy Qwen on GPU with prefill-decode disaggregation and verify chat completions."""

    def test_llmd_prefill_decode(
        self,
        llmisvc: LLMInferenceService,
    ):
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
