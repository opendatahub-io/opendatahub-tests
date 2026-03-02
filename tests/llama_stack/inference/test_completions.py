import pytest
from simple_logger.logger import get_logger
from llama_stack_client import LlamaStackClient
from tests.llama_stack.constants import ModelInfo

LOGGER = get_logger(name=__name__)


@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [
        pytest.param(
            {"name": "test-llamastack-infer-completions", "randomize_name": True},
        ),
    ],
    indirect=True,
)
@pytest.mark.llama_stack
class TestLlamaStackInferenceCompletions:
    """Test class for LlamaStack Inference API for Chat Completions and Completions

    For more information about this API, see:
    - https://llamastack.github.io/docs/references/python_sdk_reference#inference
    - https://github.com/openai/openai-python/blob/main/api.md#completions-1
    """

    @pytest.mark.smoke
    def test_inference_chat_completion(
        self,
        unprivileged_llama_stack_client: LlamaStackClient,
        llama_stack_models: ModelInfo,
    ) -> None:
        """Test chat completion functionality with a simple ACK response."""
        response = unprivileged_llama_stack_client.chat.completions.create(
            model=llama_stack_models.model_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Just respond ACK."},
            ],
            temperature=0,
        )
        assert len(response.choices) > 0, "No response after basic inference on llama-stack server"

        # Check if response has the expected structure and content
        content = response.choices[0].message.content
        assert content is not None, "LLM response content is None"
        assert "ACK" in content, "The LLM didn't provide the expected answer to the prompt"

    @pytest.mark.smoke
    def test_inference_completion(
        self,
        unprivileged_llama_stack_client: LlamaStackClient,
        llama_stack_models: ModelInfo,
    ) -> None:
        """Test text completion functionality with a geography question."""
        response = unprivileged_llama_stack_client.completions.create(
            model=llama_stack_models.model_id, prompt="What is the capital of Catalonia?", max_tokens=20, temperature=0
        )
        assert len(response.choices) > 0, "No response after basic inference on llama-stack server"

        # Check if response has the expected structure and content
        content = response.choices[0].text.lower()
        assert content is not None, "LLM response content is None"
        LOGGER.info(f"LLM response content for test_inference_completion: {content}")
        assert "barcelona" in content, "The LLM didn't provide the expected answer to the prompt"
