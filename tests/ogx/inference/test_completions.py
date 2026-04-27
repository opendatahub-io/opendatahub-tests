import pytest
import structlog
from ogx_client import OgxClient

from tests.ogx.constants import ModelInfo

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [
        pytest.param(
            {"name": "test-ogx-infer-completions", "randomize_name": True},
        ),
    ],
    indirect=True,
)
@pytest.mark.ogx
class TestOgxInferenceCompletions:
    """Test class for OGX Inference API for Chat Completions and Completions

    For more information about this API, see:
    - https://ogx-ai.github.io/docs/references/python_sdk_reference#inference
    - https://github.com/openai/openai-python/blob/main/api.md#completions-1
    """

    @pytest.mark.tier1
    def test_inference_chat_completion(
        self,
        unprivileged_ogx_client: OgxClient,
        ogx_models: ModelInfo,
    ) -> None:
        """Test chat completion functionality with a simple ACK response."""
        response = unprivileged_ogx_client.chat.completions.create(
            model=ogx_models.model_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Just respond ACK."},
            ],
            temperature=0,
        )
        assert len(response.choices) > 0, "No response after basic inference on ogx server"

        # Check if response has the expected structure and content
        content = response.choices[0].message.content
        assert content is not None, "LLM response content is None"
        assert "ACK" in content, "The LLM didn't provide the expected answer to the prompt"

    @pytest.mark.tier1
    def test_inference_completion(
        self,
        unprivileged_ogx_client: OgxClient,
        ogx_models: ModelInfo,
    ) -> None:
        """Test text completion functionality with a geography question."""
        response = unprivileged_ogx_client.completions.create(
            model=ogx_models.model_id, prompt="What is the capital of Catalonia?", max_tokens=20, temperature=0
        )
        assert len(response.choices) > 0, "No response after basic inference on ogx server"

        # Check if response has the expected structure and content
        content = response.choices[0].text.lower()
        assert content is not None, "LLM response content is None"
        LOGGER.info(f"LLM response content for test_inference_completion: {content}")
        assert "barcelona" in content, "The LLM didn't provide the expected answer to the prompt"
