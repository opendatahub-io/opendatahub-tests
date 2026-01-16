import pytest
from llama_stack_client import LlamaStackClient
from llama_stack_client.types.vector_store import VectorStore
from simple_logger.logger import get_logger
from tests.llama_stack.constants import ModelInfo
from tests.llama_stack.utils import (
    validate_api_responses,
    create_response_function,
    get_torchtune_test_expectations,
)

LOGGER = get_logger(name=__name__)


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llama_stack_server_config, vector_store",
    [
        pytest.param(
            {"name": "test-llamastack-vector-stores", "randomize_name": True},
            {
                "llama_stack_storage_size": "2Gi",
                "vector_io_provider": "milvus",
                "files_provider": "s3",
            },
            {"vector_io_provider": "milvus"},
            id="vector_io_provider_milvus+files_provider_s3",
        ),
        pytest.param(
            {"name": "test-llamastack-vector-stores", "randomize_name": True},
            {
                "llama_stack_storage_size": "2Gi",
                "vector_io_provider": "faiss",
                "files_provider": "local",
            },
            {"vector_io_provider": "faiss"},
            id="vector_io_provider_faiss+files_provider_local",
        ),
        pytest.param(
            {"name": "test-llamastack-vector-stores", "randomize_name": True},
            {
                "llama_stack_storage_size": "2Gi",
                "vector_io_provider": "milvus-remote",
                "files_provider": "s3",
            },
            {"vector_io_provider": "milvus-remote"},
            id="vector_io_provider_milvus-remote+files_provider_s3",
        ),
        pytest.param(
            {"name": "test-llamastack-vector-stores", "randomize_name": True},
            {
                "llama_stack_storage_size": "2Gi",
                "vector_io_provider": "pgvector",
            },
            {"vector_io_provider": "pgvector"},
            id="vector_io_provider_pgvector",
        ),
        pytest.param(
            {"name": "test-llamastack-vector-stores", "randomize_name": True},
            {
                "llama_stack_storage_size": "2Gi",
                "vector_io_provider": "qdrant-remote",
            },
            {"vector_io_provider": "qdrant-remote"},
            id="vector_io_provider_qdrant-remote",
        ),
    ],
    indirect=True,
)
@pytest.mark.rag
class TestLlamaStackVectorStores:
    """Test class for LlamaStack OpenAI Compatible Vector Stores API

    Note: multiple vector_io providers are tested thanks to the pytest.param vector_io_provider

    For more information about this API, see:
    - https://github.com/llamastack/llama-stack-client-python/blob/main/api.md#vectorstores
    - https://github.com/openai/openai-python/blob/main/api.md#vectorstores
    """

    @pytest.mark.smoke
    def test_vector_stores_create_search(
        self,
        unprivileged_llama_stack_client: LlamaStackClient,
        llama_stack_models: ModelInfo,
        vector_store_with_example_docs: VectorStore,
    ) -> None:
        """
        Test vector_stores and responses API

        Uses a vector store with pre-uploaded TorchTune documentation files and tests the responses API
        with file search capabilities. Validates that the API can retrieve and use
        knowledge from uploaded documents to answer questions.
        """

        _response_fn = create_response_function(
            llama_stack_client=unprivileged_llama_stack_client,
            llama_stack_models=llama_stack_models,
            vector_store=vector_store_with_example_docs,
        )

        turns_with_expectations = get_torchtune_test_expectations()

        validation_result = validate_api_responses(response_fn=_response_fn, test_cases=turns_with_expectations)

        assert validation_result["success"], f"RAG agent validation failed. Summary: {validation_result['summary']}"

    @pytest.mark.smoke
    def test_vector_stores_search(
        self,
        unprivileged_llama_stack_client: LlamaStackClient,
        vector_store_with_example_docs: VectorStore,
    ) -> None:
        """
        Test vector_stores search functionality using the search endpoint.

        Uses a vector store with pre-uploaded TorchTune documentation files and tests the search API
        to retrieve relevant chunks based on query text. Validates that the search
        returns relevant results with proper metadata and content.
        """

        search_queries = [
            "What is LoRA fine-tuning?",
            "How does quantization work?",
            "What are memory optimizations?",
            "Tell me about DoRA",
            "What is TorchTune?",
        ]

        for query in search_queries:
            # Use the vector store search endpoint
            search_response = unprivileged_llama_stack_client.vector_stores.search(
                vector_store_id=vector_store_with_example_docs.id, query=query
            )

            # Validate search response
            assert search_response is not None, f"Search response is None for query: {query}"
            assert hasattr(search_response, "data"), "Search response missing 'data' attribute"
            assert isinstance(search_response.data, list), "Search response data should be a list"

            # Check that we got some results
            assert len(search_response.data) > 0, f"No search results returned for query: {query}"

            # Validate each search result
            for result in search_response.data:
                assert hasattr(result, "content"), "Search result missing 'content' attribute"
                assert result.content is not None, "Search result content should not be None"
                assert len(result.content) > 0, "Search result content should not be empty"

        LOGGER.info(f"Successfully tested vector store search with {len(search_queries)} queries")
