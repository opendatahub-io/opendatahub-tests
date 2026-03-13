import pytest
from llama_stack_client import LlamaStackClient
from llama_stack_client.types.vector_store import VectorStore

from tests.llama_stack.constants import ModelInfo
from tests.llama_stack.utils import (
    create_response_function,
    get_torchtune_test_expectations,
    validate_api_responses,
)


def _assert_minimal_rag_response(
    unprivileged_llama_stack_client: LlamaStackClient,
    llama_stack_models: ModelInfo,
    vector_store_with_example_docs: VectorStore,
) -> None:
    response_fn = create_response_function(
        llama_stack_client=unprivileged_llama_stack_client,
        llama_stack_models=llama_stack_models,
        vector_store=vector_store_with_example_docs,
    )

    test_case = get_torchtune_test_expectations()[0]
    answer = response_fn(question=test_case["question"])

    assert answer is not None, "RAG response content is None"
    assert isinstance(answer, str), "RAG response content should be a string"
    assert answer.strip(), "RAG response content is empty"


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llama_stack_server_config, vector_store",
    [
        pytest.param(
            {"name": "test-llamastack-vector-rag-upgrade"},
            {
                "llama_stack_storage_size": "2Gi",
                "vector_io_provider": "milvus",
                "files_provider": "s3",
            },
            {"vector_io_provider": "milvus"},
        ),
    ],
    indirect=True,
)
@pytest.mark.llama_stack
@pytest.mark.rag
class TestPreUpgradeLlamaStackVectorStoreRag:
    @pytest.mark.pre_upgrade
    @pytest.mark.smoke
    def test_vector_store_rag_pre_upgrade(
        self,
        unprivileged_llama_stack_client: LlamaStackClient,
        llama_stack_models: ModelInfo,
        vector_store_with_example_docs: VectorStore,
    ) -> None:
        """Verify vector-store-backed RAG works before upgrade.

        Given: A running unprivileged LlamaStack distribution with a vector store and uploaded documents.
        When: A retrieval-augmented response is requested using file search.
        Then: The generated answer is non-empty, confirming baseline RAG behavior before upgrade.
        """
        _assert_minimal_rag_response(
            unprivileged_llama_stack_client=unprivileged_llama_stack_client,
            llama_stack_models=llama_stack_models,
            vector_store_with_example_docs=vector_store_with_example_docs,
        )


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llama_stack_server_config, vector_store",
    [
        pytest.param(
            {"name": "test-llamastack-vector-rag-upgrade"},
            {
                "llama_stack_storage_size": "2Gi",
                "vector_io_provider": "milvus",
                "files_provider": "s3",
            },
            {"vector_io_provider": "milvus"},
        ),
    ],
    indirect=True,
)
@pytest.mark.llama_stack
@pytest.mark.rag
class TestPostUpgradeLlamaStackVectorStoreRag:
    @pytest.mark.post_upgrade
    @pytest.mark.smoke
    def test_vector_store_rag_post_upgrade(
        self,
        unprivileged_llama_stack_client: LlamaStackClient,
        llama_stack_models: ModelInfo,
        vector_store_with_example_docs: VectorStore,
    ) -> None:
        """Verify vector-store-backed RAG remains correct after upgrade.

        Given: A pre-existing unprivileged LlamaStack distribution after upgrade with reused vector store docs.
        When: The RAG response flow is validated across TorchTune expectation turns.
        Then: All expectation checks pass, proving retrieval+inference continuity after upgrade.
        """
        response_fn = create_response_function(
            llama_stack_client=unprivileged_llama_stack_client,
            llama_stack_models=llama_stack_models,
            vector_store=vector_store_with_example_docs,
        )

        turns_with_expectations = get_torchtune_test_expectations()
        validation_result = validate_api_responses(
            response_fn=response_fn,
            test_cases=turns_with_expectations,
        )

        assert validation_result["success"], (
            f"Post-upgrade RAG validation failed. Summary: {validation_result['summary']}"
        )
