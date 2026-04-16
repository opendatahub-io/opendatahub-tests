import pytest
import structlog
from llama_stack_client import APIConnectionError, InternalServerError, LlamaStackClient
from llama_stack_client.types.vector_store import VectorStore

from tests.llama_stack.constants import ModelInfo
from tests.llama_stack.datasets import (
    FINANCE_DATASET,
    IBM_2025_Q4_EARNINGS,
    IBM_2025_Q4_EARNINGS_ENCRYPTED,
    Dataset,
)

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llama_stack_server_config, vector_store, dataset",
    [
        pytest.param(
            {"name": "test-llamastack-vector-stores", "randomize_name": True},
            {
                "llama_stack_storage_size": "2Gi",
                "vector_io_provider": "milvus",
                "embedding_provider": "sentence-transformers",
                "files_provider": "local",
            },
            {"vector_io_provider": "milvus", "dataset": IBM_2025_Q4_EARNINGS},
            IBM_2025_Q4_EARNINGS,
            id="vector_io:milvus, files:local, embedding:sentence-transformers, dataset:IBM_2025_Q4_EARNINGS",
            marks=(pytest.mark.smoke),
        ),
        pytest.param(
            {"name": "test-llamastack-vector-stores", "randomize_name": True},
            {
                "llama_stack_storage_size": "2Gi",
                "vector_io_provider": "milvus",
                "embedding_provider": "sentence-transformers",
                "files_provider": "local",
            },
            {"vector_io_provider": "milvus", "dataset": IBM_2025_Q4_EARNINGS_ENCRYPTED},
            IBM_2025_Q4_EARNINGS_ENCRYPTED,
            id="vector_io:milvus, files:local, embedding:sentence-transformers, dataset:IBM_2025_Q4_EARNINGS_ENCRYPTED",
            marks=(pytest.mark.tier1, pytest.mark.xfail(reason="RHAIENG-3816")),
        ),
        pytest.param(
            {"name": "test-llamastack-vector-stores", "randomize_name": True},
            {
                "vector_io_provider": "milvus-remote",
                "embedding_provider": "vllm-embedding",
                "files_provider": "s3",
            },
            {"vector_io_provider": "milvus-remote", "dataset": FINANCE_DATASET},
            FINANCE_DATASET,
            id="vector_io:milvus-remote, files: s3, embedding: vllm-embedding, dataset:FINANCE_DATASET",
            marks=(pytest.mark.smoke),
        ),
        pytest.param(
            {"name": "test-llamastack-vector-stores", "randomize_name": True},
            {
                "llama_stack_storage_size": "2Gi",
                "vector_io_provider": "faiss",
                "files_provider": "local",
            },
            {"vector_io_provider": "faiss", "dataset": FINANCE_DATASET},
            FINANCE_DATASET,
            id="vector_io: faiss, files:local, embedding: vllm-embedding, dataset:FINANCE_DATASET",
            marks=(pytest.mark.tier1),
        ),
        pytest.param(
            {"name": "test-llamastack-vector-stores", "randomize_name": True},
            {
                "llama_stack_storage_size": "2Gi",
                "vector_io_provider": "pgvector",
                "files_provider": "s3",
            },
            {"vector_io_provider": "pgvector", "dataset": FINANCE_DATASET},
            FINANCE_DATASET,
            id="vector_io:pgvector, files:s3, embedding: vllm-embedding, dataset:FINANCE_DATASET",
            marks=(pytest.mark.tier1),
        ),
        pytest.param(
            {"name": "test-llamastack-vector-stores", "randomize_name": True},
            {
                "llama_stack_storage_size": "2Gi",
                "vector_io_provider": "qdrant-remote",
                "files_provider": "local",
            },
            {"vector_io_provider": "qdrant-remote", "dataset": FINANCE_DATASET},
            FINANCE_DATASET,
            id="vector_io:qdrant-remote, files:local, embedding: vllm-embedding, dataset:FINANCE_DATASET",
            marks=(pytest.mark.tier1),
        ),
    ],
    indirect=True,
)
@pytest.mark.rag
class TestLlamaStackVectorStores:
    """Test class for LlamaStack OpenAI Compatible Vector Stores API

    Note: multiple vector_io providers and datasets are tested via pytest parametrize

    For more information about this API, see:
    - https://github.com/llamastack/llama-stack-client-python/blob/main/api.md#vectorstores
    - https://github.com/openai/openai-python/blob/main/api.md#vectorstores
    """

    def test_vector_stores_file_upload(
        self,
        unprivileged_llama_stack_client: LlamaStackClient,
        vector_store: VectorStore,
        dataset: Dataset,
    ) -> None:
        """Verify vector store files listing after document upload.

        Given: A vector store populated with the parametrized dataset documents.
        When: The vector_stores.files.list API is called with filter="completed".
        Then: The list returns exactly one completed file per uploaded document,
            confirming ingestion is visible through the OpenAI-compatible API.
        """
        store_id = vector_store.id
        completed_files = list(
            unprivileged_llama_stack_client.vector_stores.files.list(
                vector_store_id=store_id,
                filter="completed",
            )
        )
        assert len(completed_files) == len(dataset.documents), (
            f"Expected {len(dataset.documents)} completed vector store file(s) in {store_id!r} after upload, "
            f"found {len(completed_files)}"
        )
        LOGGER.info(f"Vector store {store_id} lists {len(completed_files)} completed file(s)")

    def test_vector_stores_search(
        self,
        unprivileged_llama_stack_client: LlamaStackClient,
        vector_store: VectorStore,
        dataset: Dataset,
    ) -> None:
        """
        Test vector_stores search functionality using the search endpoint.

        Given: A vector store populated with the parametrized dataset documents
        When: Queries from the dataset QA ground truth are executed per retrieval mode
        Then: Each mode returns relevant results with proper metadata and content
        """
        search_modes = sorted({r.retrieval_mode for r in dataset.load_qa()})

        provider_id = (vector_store.metadata or {}).get("provider_id", "")
        # FAISS does not support hybrid and keyword search modes see:
        # https://github.com/llamastack/llama-stack/blob/main/src/llama_stack/providers/inline/vector_io/faiss/faiss.py#L180-L196
        if provider_id == "faiss":
            search_modes = [m for m in search_modes if m == "vector"]

        for search_mode in search_modes:
            qa_records = dataset.load_qa(retrieval_mode=search_mode)
            for record in qa_records:
                search_response = unprivileged_llama_stack_client.vector_stores.search(
                    vector_store_id=vector_store.id,
                    query=record.question,
                    search_mode=search_mode,
                    max_num_results=10,
                )

                assert search_response is not None, (
                    f"Search response is None for mode={search_mode!r} query={record.question!r}"
                )
                assert hasattr(search_response, "data"), "Search response missing 'data' attribute"
                assert isinstance(search_response.data, list), "Search response data should be a list"
                assert len(search_response.data) > 0, (
                    f"No search results for mode={search_mode!r} query={record.question!r}"
                )

                for result in search_response.data:
                    assert hasattr(result, "content"), "Search result missing 'content' attribute"
                    assert result.content is not None, "Search result content should not be None"
                    assert len(result.content) > 0, "Search result content should not be empty"

            LOGGER.info(f"Search mode {search_mode!r}: {len(qa_records)} queries returned results")

        LOGGER.info(f"Successfully tested vector store search across modes: {search_modes}")

    def test_response_file_search_tool_invocation(
        self,
        unprivileged_llama_stack_client: LlamaStackClient,
        llama_stack_models: ModelInfo,
        vector_store: VectorStore,
        dataset: Dataset,
        subtests: pytest.Subtests,
    ) -> None:
        """
        Test file_search tool invocation and results via the responses API.

        Sends a question from the dataset QA ground truth to the responses API with the
        file_search tool attached to the vector_store fixture. Asserts the response contains
        a completed file_search_call with results carrying file metadata, and that the
        message includes file_citation annotations with file_id and filename.
        """
        vector_question = next(r.question for r in dataset.load_qa(retrieval_mode="vector"))

        try:
            response = unprivileged_llama_stack_client.responses.create(
                input=vector_question,
                model=llama_stack_models.model_id,
                instructions="Always use the file_search tool to look up information before answering.",
                stream=False,
                max_output_tokens=4096,
                tools=[
                    {
                        "type": "file_search",
                        "vector_store_ids": [vector_store.id],
                    }
                ],
            )

            with subtests.test(msg="file_search_call status should be completed"):
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

            with subtests.test(msg="file_citation annotations"):
                # Verify the message contains file_citation annotations
                annotations = []
                for item in response.output:
                    if item.type != "message" or not isinstance(item.content, list):
                        continue
                    for content_item in item.content:
                        if content_item.annotations:
                            annotations.extend(content_item.annotations)

                assert annotations, "Response message should contain annotations when file_search returns results"

                citation_annotations = [a for a in annotations if a.type == "file_citation"]
                assert citation_annotations, (
                    f"Expected at least one file_citation annotation, got types: {[a.type for a in annotations]}"
                )

                for annotation in citation_annotations:
                    assert annotation.file_id, "Annotation must include a non-empty file_id"
                    assert annotation.filename, "Annotation must include a non-empty filename"
                    assert annotation.index is not None, "Annotation must include an index"

                LOGGER.info(
                    f"Found {len(citation_annotations)} file_citation annotation(s). "
                    f"File IDs: {[a.file_id for a in citation_annotations]}. "
                    f"Filenames: {[a.filename for a in citation_annotations]}. "
                    f"Indexes: {[a.index for a in citation_annotations]}. "
                )

        except (APIConnectionError, InternalServerError) as exc:
            pytest.fail(f"LlamaStack server returned 500 for file_search query {vector_question!r}: {exc}")
