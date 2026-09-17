import pytest

from tests.model_serving.model_server.llmd.llmd_configs import (
    TinyLlamaHfConfig,
    TinyLlamaHfConnectionConfig,
    TinyLlamaOciConnectionConfig,
    TinyLlamaS3Config,
    TinyLlamaS3ConnectionConfig,
)
from tests.model_serving.model_server.llmd.utils import (
    ns_from_file,
    parse_completion_text,
    send_chat_completions,
    workaround_503_no_healthy_upstream,
)
from utilities.resources.llm_inference_service import LLMInferenceService

pytestmark = [pytest.mark.tier1]

NAMESPACE = ns_from_file(file=__file__)


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmisvc",
    [
        pytest.param({"name": NAMESPACE}, TinyLlamaS3Config, id="s3"),
        pytest.param({"name": NAMESPACE}, TinyLlamaHfConfig, id="hf"),
        pytest.param({"name": NAMESPACE}, TinyLlamaS3ConnectionConfig, id="s3-connection"),
        pytest.param(
            {"name": NAMESPACE},
            TinyLlamaHfConnectionConfig,
            id="hf-connection",
            marks=[pytest.mark.skip_on_disconnected],
        ),
        pytest.param({"name": NAMESPACE}, TinyLlamaOciConnectionConfig, id="oci-connection"),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("valid_aws_config")
class TestLlmdConnectionCpu:
    """Deploy TinyLlama on CPU via static and ConnectionsAPI-driven storage, verify chat completions."""

    def test_llmd_connection_cpu(self, llmisvc: LLMInferenceService):
        """Test steps:

        1. For ConnectionsAPI configs, assert the webhook injected the expected storage/SA fields
           (no-op for static configs).
        2. Send a chat completion request to /v1/chat/completions.
        3. Assert the response status is 200.
        4. Assert the completion text contains the expected answer.
        """
        llmisvc.connections_config.verify_injection(llmisvc=llmisvc)

        prompt = "What is the capital of Italy?"
        expected = "rome"

        workaround_503_no_healthy_upstream(llmisvc=llmisvc, prompt=prompt)

        status, body = send_chat_completions(llmisvc=llmisvc, prompt=prompt)
        assert status == 200, f"Expected 200, got {status}: {body}"
        completion = parse_completion_text(response_body=body)
        assert expected in completion.lower(), f"Expected '{expected}' in response, got: {completion}"
