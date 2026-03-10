import pytest
from ocp_resources.llm_inference_service import LLMInferenceService

from tests.model_serving.model_server.llmd.llmd_configs import QwenHfConfig, QwenS3Config
from tests.model_serving.model_server.llmd.utils import (
    ns_from_file,
    parse_completion_text,
    send_chat_completions,
)

pytestmark = [pytest.mark.tier1, pytest.mark.gpu]

NAMESPACE = ns_from_file(file=__file__)


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmisvc",
    [
        pytest.param({"name": NAMESPACE}, QwenS3Config, id="s3"),
        pytest.param({"name": NAMESPACE}, QwenHfConfig, id="hf"),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("valid_aws_config")
class TestLlmdConnectionGpu:
    """Deploy Qwen on GPU via S3 and HuggingFace and verify chat completions."""

    def test_llmd_connection_gpu(
        self,
        llmisvc: LLMInferenceService,
        gpu_count_on_cluster: int,
    ):
        """Test steps:

        1. Skip if no GPU nodes are available on the cluster.
        2. Send a chat completion request to /v1/chat/completions.
        3. Assert the response status is 200.
        4. Assert the completion text contains the expected answer.
        """
        if gpu_count_on_cluster < 1:
            pytest.skip("No GPUs available on cluster, skipping GPU test")

        prompt = "What is the capital of Italy?"
        expected = "rome"

        status, body = send_chat_completions(llmisvc=llmisvc, prompt=prompt)
        assert status == 200, f"Expected 200, got {status}: {body}"
        completion = parse_completion_text(response_body=body)
        assert expected in completion.lower(), f"Expected '{expected}' in response, got: {completion}"
