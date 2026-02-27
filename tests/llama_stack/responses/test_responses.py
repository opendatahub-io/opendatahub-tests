import pytest
from llama_stack_client import LlamaStackClient

from tests.llama_stack.constants import ModelInfo


@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [
        pytest.param(
            {"name": "test-llamastack-responses", "randomize_name": True},
        ),
    ],
    indirect=True,
)
@pytest.mark.rag
@pytest.mark.skip_must_gather
class TestLlamaStackResponses:
    """Test class for LlamaStack responses API functionality.

    For more information about this API, see:
    - https://github.com/llamastack/llama-stack-client-python/blob/main/api.md#responses
    - https://github.com/openai/openai-python/blob/main/api.md#responses
    """

    @pytest.mark.smoke
    def test_responses_create(
        self,
        unprivileged_llama_stack_client: LlamaStackClient,
        llama_stack_models: ModelInfo,
    ) -> None:
        """
        Test simple responses API from the llama-stack server.

        Validates basic text generation capabilities using the responses API endpoint.
        Tests identity and capability questions to ensure the LLM can provide
        appropriate responses about itself and its functionality.
        """
        test_cases = [
            ("Who are you?", ["model", "assistant", "ai", "artificial", "language model"]),
            ("What can you do?", ["answer"]),
        ]

        for question, expected_keywords in test_cases:
            response = unprivileged_llama_stack_client.responses.create(
                model=llama_stack_models.model_id,
                input=question,
                instructions="You are a helpful assistant.",
            )

            content = response.output_text
            assert content is not None, "LLM response content is None"
            assert any(keyword in content.lower() for keyword in expected_keywords), (
                f"The LLM didn't provide any of the expected keywords {expected_keywords}. Got: {content}"
            )
