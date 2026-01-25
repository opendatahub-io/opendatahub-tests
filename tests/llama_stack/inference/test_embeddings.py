import pytest
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import CreateEmbeddingsResponse

from tests.llama_stack.constants import ModelInfo


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llama_stack_server_config",
    [
        pytest.param(
            {"name": "test-llamastack-infer-embeddings", "randomize_name": True},
            {"embedding_provider": "vllm-embedding"},
            id="embedding_provider_vllm-embedding",
        ),
        pytest.param(
            {"name": "test-llamastack-infer-embeddings", "randomize_name": True},
            {"embedding_provider": "sentence-transformers"},
            id="embedding_provider_sentence-transformers",
        ),
    ],
    indirect=True,
)
@pytest.mark.llama_stack
class TestLlamaStackInferenceEmbeddings:
    """Test class for LlamaStack Inference API for Embeddings

    This test suite is parametrized to test:
    - A remote embeddings model served with vllm
    - The embedding model included in the Red Hat Llama Stack Distribution,
      ibm-granite/granite-embedding-125m-english, served with sentence-transformers

    For more information about this API, see:
    - https://llamastack.github.io/docs/references/python_sdk_reference#inference
    - https://github.com/openai/openai-python/blob/main/api.md#embeddings
    """

    @pytest.mark.smoke
    def test_inference_embeddings(
        self,
        llama_stack_models: ModelInfo,
        unprivileged_llama_stack_client: LlamaStackClient,
    ) -> None:
        """
        Test embedding model functionality and vector generation.

        Validates that the server can generate properly formatted embedding vectors
        for text input with correct dimensions as specified in model metadata.
        """

        # Embed single input text with encoding_format=float (the returned embedding item is a list of floats)
        embeddings_response = unprivileged_llama_stack_client.embeddings.create(
            model=llama_stack_models.embedding_model.id,
            input="The food was delicious and the waiter...",
            encoding_format="float",
        )
        assert isinstance(embeddings_response, CreateEmbeddingsResponse)
        assert len(embeddings_response.data) == 1
        assert isinstance(embeddings_response.data[0].embedding, list)
        assert llama_stack_models.embedding_dimension == len(embeddings_response.data[0].embedding)
        assert isinstance(embeddings_response.data[0].embedding[0], float)

        # Embed single input text with encoding_format=base64  (the returned embedding item is
        # a single base64-encoded string)
        embeddings_response = unprivileged_llama_stack_client.embeddings.create(
            model=llama_stack_models.embedding_model.id,
            input="The food was delicious and the waiter...",
            encoding_format="base64",
        )
        assert isinstance(embeddings_response, CreateEmbeddingsResponse)
        assert len(embeddings_response.data) == 1
        assert isinstance(embeddings_response.data[0].embedding, str)

        # Embed multiple input sets with encoding_format=float (each returned embedding item is a list of floats)
        input_list = ["Input text 1", "Input text 1", "Input text 1"]
        embeddings_response = unprivileged_llama_stack_client.embeddings.create(
            model=llama_stack_models.embedding_model.id, input=input_list, encoding_format="float"
        )
        assert isinstance(embeddings_response, CreateEmbeddingsResponse)
        assert len(embeddings_response.data) == len(input_list)
        for item in range(len(input_list)):
            assert isinstance(embeddings_response.data[item].embedding, list)
            assert llama_stack_models.embedding_dimension == len(embeddings_response.data[item].embedding)
            assert isinstance(embeddings_response.data[item].embedding[0], float)

        # Embed multiple input sets with base64 encoding format (each returned embedding a single base64-encoded string)
        input_list = ["Input text 1", "Input text 1", "Input text 1"]
        embeddings_response = unprivileged_llama_stack_client.embeddings.create(
            model=llama_stack_models.embedding_model.id, input=input_list, encoding_format="base64"
        )
        assert isinstance(embeddings_response, CreateEmbeddingsResponse)
        assert len(embeddings_response.data) == len(input_list)
        for item in range(len(input_list)):
            assert isinstance(embeddings_response.data[item].embedding, str)
