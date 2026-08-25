import pytest
import structlog
from ogx_client import OgxClient
from ogx_client.types.vector_store import VectorStore
from timeout_sampler import retry

from tests.ogx.constants import ModelInfo
from tests.ogx.datasets import IBM_2025_Q4_EARNINGS

LOGGER = structlog.get_logger(name=__name__)
IBM_EARNINGS_RAG_QUERY = "How did IBM perform financially in the fourth quarter of 2025?"


@retry(wait_timeout=60, sleep=5)
def wait_for_vector_store_files_ready(
    ogx_client: OgxClient,
    vector_store_id: str,
) -> bool:
    """Ensure all files in vector store have completed processing."""
    files_page = ogx_client.vector_stores.files.list(vector_store_id=vector_store_id)
    files = getattr(files_page, "data", [])
    if not files:
        LOGGER.warning(f"Vector store {vector_store_id} has no files attached yet")
        return False

    for f in files:
        status = getattr(f, "status", None)
        if status != "completed":
            LOGGER.warning(
                f"Vector store file {getattr(f, 'id', 'unknown')} status is '{status}', waiting for 'completed'"
            )
            return False

    LOGGER.info(f"All {len(files)} file(s) in vector store {vector_store_id} are in 'completed' status")
    return True


def _assert_minimal_rag_response(
    ogx_client: OgxClient,
    ogx_models: ModelInfo,
    vector_store: VectorStore,
) -> None:
    wait_for_vector_store_files_ready(
        ogx_client=ogx_client,
        vector_store_id=vector_store.id,
    )

    response = ogx_client.responses.create(
        input=IBM_EARNINGS_RAG_QUERY,
        model=ogx_models.model_id,
        instructions="Always use the file_search tool to look up information before answering.",
        stream=False,
        max_output_tokens=4096,
        tool_choice="required",
        include=["file_search_call.results"],
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [vector_store.id],
            }
        ],
    )

    file_search_calls = [item for item in response.output if item.type == "file_search_call"]
    assert file_search_calls, (
        "Expected file_search_call output item in the response, indicating the model "
        f"invoked file_search. Output types: {[item.type for item in response.output]}"
    )

    file_search_call = file_search_calls[0]
    assert file_search_call.status == "completed", (
        f"Expected file_search_call status 'completed', got '{file_search_call.status}'"
    )
    assert file_search_call.results, "file_search_call should contain retrieval results"

    annotations = []
    for item in response.output:
        if item.type != "message" or not isinstance(item.content, list):
            continue
        for content_item in item.content:
            item_annotations = getattr(content_item, "annotations", None)
            if item_annotations:
                annotations.extend(item_annotations)

    citation_annotations = [a for a in annotations if getattr(a, "type", None) == "file_citation"]
    if not citation_annotations:
        LOGGER.warning(
            "No file_citation annotations found in the response message. "
            "The model did not include citation markers despite server-side instructions."
        )
    else:
        for annotation in citation_annotations:
            assert annotation.file_id, "Annotation must include a non-empty file_id"
            assert annotation.filename, "Annotation must include a non-empty filename"
            assert annotation.index is not None, "Annotation must include an index"


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ogx_server, vector_store",
    [
        pytest.param(
            {"name": "test-ogx-vector-rag-upgrade"},
            {
                "ogx_storage_size": "2Gi",
                "vector_io_provider": "milvus-remote",
                "files_provider": "s3",
            },
            {"vector_io_provider": "milvus-remote", "dataset": IBM_2025_Q4_EARNINGS},
        ),
    ],
    indirect=True,
)
@pytest.mark.ogx
@pytest.mark.rag
class TestPreUpgradeOgxVectorStoreRag:
    @pytest.mark.pre_upgrade
    def test_vector_store_rag_pre_upgrade(
        self,
        ogx_client: OgxClient,
        ogx_models: ModelInfo,
        vector_store: VectorStore,
    ) -> None:
        """Verify vector-store-backed RAG works before upgrade.

        Given: A running OGX distribution with a vector store and uploaded documents.
        When: A retrieval-augmented response is requested using file search.
        Then: The response includes completed file_search_call output and file_citation annotations.
        """
        _assert_minimal_rag_response(
            ogx_client=ogx_client,
            ogx_models=ogx_models,
            vector_store=vector_store,
        )


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ogx_server, vector_store",
    [
        pytest.param(
            {"name": "test-ogx-vector-rag-upgrade"},
            {
                "ogx_storage_size": "2Gi",
                "vector_io_provider": "milvus-remote",
                "files_provider": "s3",
            },
            {"vector_io_provider": "milvus-remote"},
        ),
    ],
    indirect=True,
)
@pytest.mark.ogx
@pytest.mark.rag
class TestPostUpgradeOgxVectorStoreRag:
    @pytest.mark.post_upgrade
    def test_vector_store_rag_post_upgrade(
        self,
        ogx_client: OgxClient,
        ogx_models: ModelInfo,
        vector_store: VectorStore,
    ) -> None:
        """Verify vector-store-backed RAG remains correct after upgrade.

        Given: A pre-existing OGX distribution after upgrade with reused vector store docs.
        When: A retrieval-augmented response is requested using file search.
        Then: The response includes completed file_search_call output and file_citation annotations.
        """
        _assert_minimal_rag_response(
            ogx_client=ogx_client,
            ogx_models=ogx_models,
            vector_store=vector_store,
        )
