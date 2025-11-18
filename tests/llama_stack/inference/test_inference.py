import pytest
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import CreateEmbeddingsResponse
from tests.llama_stack.constants import ModelInfo


@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [
        pytest.param(
            {"name": "test-llamastack-inference", "randomize_name": True},
        ),
    ],
    indirect=True,
)
@pytest.mark.llama_stack
class TestLlamaStackInference:
    """Test class for LlamaStack Inference API (chat_completion, completion and embeddings)

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
            model=llama_stack_models.model_id, prompt="What is the capital of Catalonia?", max_tokens=7, temperature=0
        )
        assert len(response.choices) > 0, "No response after basic inference on llama-stack server"

        # Check if response has the expected structure and content
        content = response.choices[0].text.lower()
        assert content is not None, "LLM response content is None"
        assert "barcelona" in content, "The LLM didn't provide the expected answer to the prompt"

    @pytest.mark.smoke
    def test_inference_embeddings(
        self,
        unprivileged_llama_stack_client: LlamaStackClient,
        llama_stack_models: ModelInfo,
    ) -> None:
        """
        Test embedding model functionality and vector generation.

        Validates that the server can generate properly formatted embedding vectors
        for text input with correct dimensions as specified in model metadata.
        """

        embeddings_response = unprivileged_llama_stack_client.embeddings.create(
            model=llama_stack_models.embedding_model.identifier,
            input="The food was delicious and the waiter...",
            encoding_format="float",
        )

        assert isinstance(embeddings_response, CreateEmbeddingsResponse)
        assert len(embeddings_response.data) == 1
        assert isinstance(embeddings_response.data[0].embedding, list)
        assert llama_stack_models.embedding_dimension == len(embeddings_response.data[0].embedding)
        assert isinstance(embeddings_response.data[0].embedding[0], float)

        input_list = ["Input text 1", "Input text 1", "Input text 1"]
        embeddings_response = unprivileged_llama_stack_client.embeddings.create(
            model=llama_stack_models.embedding_model.identifier, input=input_list, encoding_format="float"
        )

        assert isinstance(embeddings_response, CreateEmbeddingsResponse)
        assert len(embeddings_response.data) == len(input_list)
        for item in range(len(input_list)):
            assert isinstance(embeddings_response.data[item].embedding, list)
            assert llama_stack_models.embedding_dimension == len(embeddings_response.data[item].embedding)
            assert isinstance(embeddings_response.data[item].embedding[0], float)
