"""CPU connection tests for LLMInferenceService (S3 + HuggingFace)."""

import pytest
from ocp_resources.llm_inference_service import LLMInferenceService

from tests.model_serving.model_server.llmd_v2.llmd_configs import Opt125mHfConfig, TinyLlamaS3Config
from tests.model_serving.model_server.llmd_v2.utils import (
    ns_from_file,
    parse_completion_text,
    send_chat_completions,
)

pytestmark = [pytest.mark.tier1, pytest.mark.llmd_cpu]

NAMESPACE = ns_from_file(__file__)


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmisvc",
    [
        pytest.param({"name": NAMESPACE}, TinyLlamaS3Config, id="s3"),
        pytest.param({"name": NAMESPACE}, Opt125mHfConfig, id="hf"),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("valid_aws_config")
class TestLlmdConnectionCpu:
    """Verify CPU inference with S3 and HuggingFace storage backends."""

    def test_llmd_connection_cpu(self, llmisvc: LLMInferenceService):
        prompt = "What is the capital of Italy?"
        expected = "rome"

        status, body = send_chat_completions(llmisvc, prompt=prompt)
        assert status == 200, f"Expected 200, got {status}: {body}"
        completion = parse_completion_text(body)
        assert expected in completion.lower(), f"Expected '{expected}' in response, got: {completion}"
