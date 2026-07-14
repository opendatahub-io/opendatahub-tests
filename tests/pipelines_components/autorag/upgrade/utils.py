import structlog
from ogx_client import OgxClient
from ogx_client.types.vector_store import VectorStore

from tests.pipelines_components.automl.upgrade.utils import (  # noqa: NIT001
    load_baseline_from_configmap,
    save_baseline_to_configmap,
)

LOGGER = structlog.get_logger(name=__name__)

UPGRADE_BASELINE_CONFIGMAP = "autorag-upgrade-baseline"

RAG_QUERY = "What is the content of the document?"

__all__ = [
    "RAG_QUERY",
    "UPGRADE_BASELINE_CONFIGMAP",
    "assert_rag_query_works",
    "discover_vector_store_ids",
    "load_baseline_from_configmap",
    "save_baseline_to_configmap",
]


def discover_vector_store_ids(ogx_client: OgxClient) -> list[str]:
    """List vector stores in OGX and return their IDs.

    The AutoRAG pipeline creates vector stores as part of the optimization.
    This discovers them so they can be saved in the baseline ConfigMap and
    queried post-upgrade.
    """
    vector_stores = ogx_client.vector_stores.list()
    ids = [vs.id for vs in vector_stores.data]
    LOGGER.info("Discovered vector stores in OGX", count=len(ids), ids=ids)
    return ids


def assert_rag_query_works(
    ogx_client: OgxClient,
    model_id: str,
    vector_store: VectorStore,
) -> None:
    """Send a RAG query via OGX Responses API and verify the response."""
    response = ogx_client.responses.create(
        input=RAG_QUERY,
        model=model_id,
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

    assert annotations, "Response should contain file_citation annotations when file_search returns results"
    assert any(annotation.type == "file_citation" for annotation in annotations), (
        "Expected at least one file_citation annotation in response output"
    )
