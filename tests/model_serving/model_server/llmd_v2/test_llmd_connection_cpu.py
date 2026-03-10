import pytest
from ocp_resources.llm_inference_service import LLMInferenceService

from tests.model_serving.model_server.llmd_v2.llmd_configs import TinyLlamaHfConfig, TinyLlamaS3Config
from tests.model_serving.model_server.llmd_v2.utils import (
    ns_from_file,
    parse_completion_text,
    send_chat_completions,
)

pytestmark = [pytest.mark.tier1, pytest.mark.cpu]

NAMESPACE = ns_from_file(__file__)


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmisvc",
    [
        pytest.param({"name": NAMESPACE}, TinyLlamaS3Config, id="s3"),
        pytest.param({"name": NAMESPACE}, TinyLlamaHfConfig, id="hf"),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("valid_aws_config")
class TestLlmdConnectionCpu:
    """Deploy TinyLlama on CPU via S3 and HuggingFace and verify chat completions."""

    def test_llmd_connection_cpu(self, llmisvc: LLMInferenceService):
        """Test steps:

        1. Send a chat completion request to /v1/chat/completions.
        2. Assert the response status is 200.
        3. Assert the completion text contains the expected answer.
        """
        prompt = "What is the capital of Italy?"
        expected = "rome"

        status, body = send_chat_completions(llmisvc, prompt=prompt)
        assert status == 200, f"Expected 200, got {status}: {body}"
        completion = parse_completion_text(body)
        assert expected in completion.lower(), f"Expected '{expected}' in response, got: {completion}"