import pytest
from ocp_resources.llm_inference_service import LLMInferenceService

from tests.model_serving.model_server.llmd.llmd_configs import TinyLlamaHfGpuConfig, TinyLlamaS3GpuConfig
from tests.model_serving.model_server.llmd.utils import (
    ns_from_file,
    parse_completion_text,
    send_chat_completions,
    workaround_503_no_healthy_upstream,
)

pytestmark = [pytest.mark.tier1, pytest.mark.gpu]

NAMESPACE = ns_from_file(file=__file__)


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmisvc",
    [
        pytest.param({"name": NAMESPACE}, TinyLlamaS3GpuConfig, id="s3"),
        pytest.param({"name": NAMESPACE}, TinyLlamaHfGpuConfig, id="hf"),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("valid_aws_config", "skip_if_no_gpu_available", "skip_if_disconnected")
class TestLlmdConnectionGpu:
    """Deploy TinyLlama on GPU via S3 and HuggingFace and verify chat completions."""

    def test_llmd_connection_gpu(
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

        workaround_503_no_healthy_upstream(llmisvc=llmisvc, prompt=prompt)

        status, body = send_chat_completions(llmisvc=llmisvc, prompt=prompt)
        assert status == 200, f"Expected 200, got {status}: {body}"
        completion = parse_completion_text(response_body=body)
        assert expected in completion.lower(), f"Expected '{expected}' in response, got: {completion}"
