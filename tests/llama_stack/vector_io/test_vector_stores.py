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


IBM_EARNINGS_SEARCH_QUERIES_BY_MODE: dict[str, list[str]] = {
    "vector": [
        "How did IBM perform financially in the fourth quarter of 2025?",
        "What were the main drivers of revenue growth?",
        "What is the company outlook for 2026?",
        "How did profit margins change year over year?",
        "What did leadership say about generative AI and growth?",
    ],
    "keyword": [
        "What was free cash flow in the fourth quarter?",
        "What was Consulting revenue and segment profit margin?",
        "What was Software revenue and constant currency growth?",
        "What was diluted earnings per share for continuing operations?",
        "What are full-year 2026 expectations for revenue and free cash flow?",
    ],
    "hybrid": [
        "What was IBM free cash flow and what does the company expect for 2026?",
        "What were segment results for Software and Infrastructure revenue?",
        "What was GAAP gross profit margin and pre-tax income?",
        "What did James Kavanaugh say about 2025 results and 2026 prospects?",
        "What was Consulting revenue and segment profit margin?",
    ],
}


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

        Iterates over vector, keyword, and hybrid search modes using
        IBM_EARNINGS_SEARCH_QUERIES_BY_MODE. Validates that each mode returns
        relevant results with proper metadata and content.
        """

        provider_id = vector_store_with_example_docs.metadata.get("provider_id", "")
        # FAISS does not support hybrid and keyword search modes see:
        # https://github.com/llamastack/llama-stack/blob/main/src/llama_stack/providers/inline/vector_io/faiss/faiss.py#L180-L196
        search_modes = ["vector"] if provider_id == "faiss" else list(IBM_EARNINGS_SEARCH_QUERIES_BY_MODE)

        for search_mode in search_modes:
            queries = IBM_EARNINGS_SEARCH_QUERIES_BY_MODE[search_mode]
            for query in queries:
                search_response = unprivileged_llama_stack_client.vector_stores.search(
                    vector_store_id=vector_store_with_example_docs.id,
                    query=query,
                    search_mode=search_mode,
                    max_num_results=10,
                )

                assert search_response is not None, f"Search response is None for mode={search_mode!r} query={query!r}"
                assert hasattr(search_response, "data"), "Search response missing 'data' attribute"
                assert isinstance(search_response.data, list), "Search response data should be a list"
                assert len(search_response.data) > 0, f"No search results for mode={search_mode!r} query={query!r}"

                for result in search_response.data:
                    assert hasattr(result, "content"), "Search result missing 'content' attribute"
                    assert result.content is not None, "Search result content should not be None"
                    assert len(result.content) > 0, "Search result content should not be empty"

            LOGGER.info(f"Search mode {search_mode!r}: {len(queries)} queries returned results")

        LOGGER.info(f"Successfully tested vector store search across modes: {search_modes}")

    @pytest.mark.smoke
    def test_response_file_search_tool_invocation(
        self,
        unprivileged_llama_stack_client: LlamaStackClient,
        llama_stack_models: ModelInfo,
        vector_store_with_example_docs: VectorStore,
    ) -> None:
        """
        Test that the file_search tool is invoked and returns results via the responses API.

        Given: A vector store with pre-uploaded documentation files
        When: A question requiring document knowledge is asked with the file_search tool
        Then: The response contains a file_search_call output with status 'completed',
              results with file_id and filename, and message annotations with file citations
        """
        response = unprivileged_llama_stack_client.responses.create(
            input=IBM_EARNINGS_SEARCH_QUERIES_BY_MODE["vector"][0],
            model=llama_stack_models.model_id,
            instructions="Always use the file_search tool to look up information before answering.",
            stream=False,
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [vector_store_with_example_docs.id],
                }
            ],
        )

        # Verify file_search_call output exists (model invoked the file_search tool)
        file_search_calls = [item for item in response.output if item.type == "file_search_call"]
        assert file_search_calls, (
            "Expected a file_search_call output item in the response, indicating the model "
            f"invoked the file_search tool. Output types: {[item.type for item in response.output]}"
        )

        file_search_call = file_search_calls[0]
        assert file_search_call.status == "completed", (
            f"Expected file_search_call status 'completed', got '{file_search_call.status}'"
        )

        # Verify file_search returned results with file metadata
        assert file_search_call.results, "file_search_call should contain search results"

        for result in file_search_call.results:
            assert result.file_id, "Search result must include a non-empty file_id"
            assert result.filename, "Search result must include a non-empty filename"
            assert result.text, "Search result must include non-empty text content"

        LOGGER.info(
            f"file_search_call returned {len(file_search_call.results)} result(s). "
            f"Filenames: {[r.filename for r in file_search_call.results]}"
        )

        # Verify the message contains file_citation annotations
        annotations = []
        for item in response.output:
            if item.type != "message" or not isinstance(item.content, list):
                continue
            for content_item in item.content:
                if content_item.annotations:
                    annotations.extend(content_item.annotations)

        assert annotations, "Response message should contain file_citation annotations when file_search returns results"

        for annotation in annotations:
            assert annotation.type == "file_citation", (
                f"Expected annotation type 'file_citation', got '{annotation.type}'"
            )
            assert annotation.file_id, "Annotation must include a non-empty file_id"
            assert annotation.filename, "Annotation must include a non-empty filename"
            assert annotation.index is not None, "Annotation must include an index"

        LOGGER.info(
            f"Found {len(annotations)} file_citation annotation(s). "
            f"File IDs: {[a.file_id for a in annotations]}. "
            f"Filenames: {[a.filename for a in annotations]}. "
            f"Indexes: {[a.index for a in annotations]}. "
        )
