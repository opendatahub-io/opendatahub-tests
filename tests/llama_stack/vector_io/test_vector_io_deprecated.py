import uuid

import pytest
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import QueryChunksResponse
from llama_stack_client.types.vector_io_insert_params import Chunk
from simple_logger.logger import get_logger
from tests.llama_stack.constants import ModelInfo

LOGGER = get_logger(name=__name__)


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llama_stack_server_config",
    [
        pytest.param(
            {"name": "test-llamastack-vector-io", "randomize_name": True},
            {"llama_stack_storage_size": "2Gi"},
        ),
    ],
    indirect=True,
)
@pytest.mark.team_rag
class TestLlamaStackVectorIODeprecated:
    """Test class for LlamaStack VectorIO API (VectorIO and VectorDBs)

    Deprecation notice: these APIs are deprecated and have been replaced
    by the OpenAI vector-stores compatible APIs (see tests in test_vector_stores.py)

    More info at "Deprecate VectorIO and VectorDBs APIs":
    - https://github.com/llamastack/llama-stack/issues/2981

    For more information about this API, see:
    - https://llamastack.github.io/docs/references/python_sdk_reference#vectorio
    - https://llamastack.github.io/docs/references/python_sdk_reference#vectordbs
    """

    @pytest.mark.smoke
    def test_vector_io_register_insert_query_unregister(
        self,
        unprivileged_llama_stack_client: LlamaStackClient,
        llama_stack_models: ModelInfo,
    ) -> None:
        """
        Validates basic vector_dbs/vector_io APIs in llama-stack using milvus:
            vector_dbs.register
            vector_dbs.unregister
            vector_io.insert
            vector_io.query

        Based on the example available at
        https://llama-stack.readthedocs.io/en/latest/building_applications/rag.html
        """
        vector_db_id = None
        try:
            vector_db = f"my-test-vector_db-{uuid.uuid4().hex}"
            res = unprivileged_llama_stack_client.vector_dbs.register(
                vector_db_id=vector_db,
                embedding_model=llama_stack_models.embedding_model.identifier,
                embedding_dimension=llama_stack_models.embedding_dimension,
                provider_id="milvus",
            )
            vector_db_id = res.identifier

            # Calculate embeddings
            embeddings_response = unprivileged_llama_stack_client.inference.embeddings(
                model_id=llama_stack_models.embedding_model.identifier,
                contents=["First chunk of text"],
                output_dimension=llama_stack_models.embedding_dimension,
            )

            # Insert chunk into the vector db
            chunks_with_embeddings = [
                Chunk(
                    content="First chunk of text",
                    mime_type="text/plain",
                    metadata={"document_id": "doc1", "source": "precomputed"},
                    embedding=embeddings_response.embeddings[0],
                ),
            ]
            unprivileged_llama_stack_client.vector_io.insert(vector_db_id=vector_db_id, chunks=chunks_with_embeddings)

            # Query the vector db to find the chunk
            chunks_response = unprivileged_llama_stack_client.vector_io.query(
                vector_db_id=vector_db_id, query="What do you know about..."
            )
            assert isinstance(chunks_response, QueryChunksResponse)
            assert len(chunks_response.chunks) > 0
            assert chunks_response.chunks[0].metadata["document_id"] == "doc1"
            assert chunks_response.chunks[0].metadata["source"] == "precomputed"

        finally:
            # Cleanup: unregister the vector database to prevent resource leaks
            if vector_db_id is not None:
                try:
                    unprivileged_llama_stack_client.vector_dbs.unregister(vector_db_id)
                except Exception as e:
                    LOGGER.warning(f"Failed to unregister vector database {vector_db_id}: {e}")
