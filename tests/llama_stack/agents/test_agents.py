import uuid
import pytest
from llama_stack_client import Agent, LlamaStackClient, RAGDocument
from simple_logger.logger import get_logger
from tests.llama_stack.constants import ModelInfo
from tests.llama_stack.utils import get_torchtune_test_expectations, validate_rag_agent_responses

LOGGER = get_logger(name=__name__)


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llama_stack_server_config",
    [
        pytest.param(
            {"name": "test-llamastack-agents", "randomize_name": True},
            {"llama_stack_storage_size": "2Gi"},
        ),
    ],
    indirect=True,
)
@pytest.mark.team_rag
class TestLlamaStackAgents:
    """Test class for LlamaStack Agents API

    For more information about this API, see:
    - https://llamastack.github.io/docs/building_applications/agent
    - https://llamastack.github.io/docs/references/python_sdk_reference#agents
    - https://llamastack.github.io/docs/building_applications/responses_vs_agents
    """

    @pytest.mark.smoke
    def test_agents_simple_agent(
        self,
        unprivileged_llama_stack_client: LlamaStackClient,
        llama_stack_models: ModelInfo,
    ) -> None:
        """
        Test basic agent creation and conversation capabilities.

        Validates agent creation (agent.create), session management (agent.create_session),
        and turn-based interactions (agent.create_turn)

        Based on "Build a Simple Agent" example available at
        https://llamastack.github.io/docs/getting_started/detailed_tutorial
        """
        agent = Agent(
            client=unprivileged_llama_stack_client,
            model=llama_stack_models.model_id,
            instructions="You are a helpful assistant.",
        )
        s_id = agent.create_session(session_name=f"s{uuid.uuid4().hex}")

        # Test identity question
        response = agent.create_turn(
            messages=[{"role": "user", "content": "Who are you?"}],
            session_id=s_id,
            stream=False,
        )
        content = response.output_message.content
        text = str(content or "")
        assert text, "LLM response content is empty"
        assert "model" in text.lower(), "The LLM didn't provide the expected answer to the prompt"

        # Test capability question
        response = agent.create_turn(
            messages=[{"role": "user", "content": "What can you do?"}],
            session_id=s_id,
            stream=False,
        )
        content = response.output_message.content
        text = str(content or "")
        assert text, "LLM response content is empty"
        assert "answer" in text.lower(), "The LLM didn't provide the expected answer to the prompt"

    @pytest.mark.smoke
    def test_agents_rag_agent(
        self,
        unprivileged_llama_stack_client: LlamaStackClient,
        llama_stack_models: ModelInfo,
    ) -> None:
        """
        Test RAG agent that can answer questions about the Torchtune project using the documents
        in a vector database.

        Creates a RAG agent with PyTorch torchtune documentation, tests knowledge queries
        about fine-tuning techniques (LoRA, QAT, memory optimizations), and validates
        that responses contain expected technical keywords.

        Based on "Build a RAG Agent" example available at
        https://llamastack.github.io/docs/getting_started/detailed_tutorial

        # TODO: update this example to use the vector_store API
        """

        vector_db_id: str | None = None
        try:
            vector_db = f"my-test-vector_db-{uuid.uuid4().hex}"
            res = unprivileged_llama_stack_client.vector_dbs.register(
                vector_db_id=vector_db,
                embedding_model=llama_stack_models.embedding_model.identifier,
                embedding_dimension=llama_stack_models.embedding_dimension,
                provider_id="milvus",
            )
            vector_db_id = res.identifier

            # Create the RAG agent connected to the vector database
            rag_agent = Agent(
                client=unprivileged_llama_stack_client,
                model=llama_stack_models.model_id,
                instructions="You are a helpful assistant. Use the RAG tool to answer questions as needed.",
                tools=[
                    {
                        "name": "builtin::rag/knowledge_search",
                        "args": {"vector_db_ids": [vector_db_id]},
                    }
                ],
            )
            session_id = rag_agent.create_session(session_name=f"s{uuid.uuid4().hex}")

            # Insert into the vector database example documents about torchtune
            urls = [
                "llama3.rst",
                "chat.rst",
                "lora_finetune.rst",
                "qat_finetune.rst",
                "memory_optimizations.rst",
            ]
            documents = [
                RAGDocument(
                    document_id=f"num-{i}",
                    content=f"https://raw.githubusercontent.com/pytorch/torchtune/refs/tags/v0.6.1/docs/source/tutorials/{url}",  # noqa
                    mime_type="text/plain",
                    metadata={},
                )
                for i, url in enumerate(urls)
            ]

            unprivileged_llama_stack_client.tool_runtime.rag_tool.insert(
                documents=documents,
                vector_db_id=vector_db_id,
                chunk_size_in_tokens=512,
            )

            turns_with_expectations = get_torchtune_test_expectations()

            # Ask the agent about the inserted documents and validate responses
            validation_result = validate_rag_agent_responses(
                rag_agent=rag_agent,
                session_id=session_id,
                turns_with_expectations=turns_with_expectations,
                stream=True,
                verbose=True,
                min_keywords_required=1,
                print_events=False,
            )

            # Assert that validation was successful
            assert validation_result["success"], f"RAG agent validation failed. Summary: {validation_result['summary']}"

            # Additional assertions for specific requirements
            for result in validation_result["results"]:
                assert result["event_count"] > 0, f"No events generated for question: {result['question']}"
                assert result["response_length"] > 0, f"No response content for question: {result['question']}"
                assert len(result["found_keywords"]) > 0, (
                    f"No expected keywords found in response for: {result['question']}"
                )

        finally:
            # Cleanup: unregister the vector database to prevent resource leaks
            if vector_db_id:
                try:
                    unprivileged_llama_stack_client.vector_dbs.unregister(vector_db_id)
                except Exception as exc:
                    LOGGER.warning("Failed to unregister vector database %s: %s", vector_db_id, exc)
