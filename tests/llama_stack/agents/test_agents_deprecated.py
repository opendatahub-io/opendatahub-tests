import uuid
import pytest
from llama_stack_client import Agent, LlamaStackClient
from llama_stack_client.types.vector_store import VectorStore
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
@pytest.mark.rag
@pytest.mark.skip_must_gather
class TestLlamaStackAgentsDeprecated:
    """Test class for LlamaStack Agents API (Deprecated)

    Deprecation Notice: The LlamaStack Agents API was removed server-side in llama-stack 0.3.0.
    It is partially implemented in llama-stack-client using the Responses API
    (https://github.com/llamastack/llama-stack-client-python/pull/281).

    Users are encouraged to use the Responses API directly.

    For more information, see:
    - https://llamastack.github.io/docs/api-deprecated/agents
    - "Migrating from Agent objects to Responses in Llama Stack":
      https://github.com/opendatahub-io/agents/blob/5902bef12c25281eecfcd3d25654de8b02857e33/migration/legacy-agents/responses-api-agent-migration.ipynb
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

        # Test expected identity indicators
        response = agent.create_turn(
            messages=[{"role": "user", "content": "Who are you?"}],
            session_id=s_id,
            stream=False,
        )
        content = response.output_text
        text = str(content or "")
        assert text, "LLM response content is empty"
        identity_indicators = ["model", "assistant", "ai", "artificial intelligence", "language model", "llm"]
        found_indicators = [indicator for indicator in identity_indicators if indicator in text.lower()]
        if not found_indicators:
            pytest.fail(
                f"Expected response to contain identity indicators like 'model', 'assistant', 'ai', etc. "
                f"Actual response: '{text[:100]}{'...' if len(text) > 100 else ''}'"
            )

        # Check expected capability indicators
        response = agent.create_turn(
            messages=[{"role": "user", "content": "What can you do?"}],
            session_id=s_id,
            stream=False,
        )
        content = response.output_text
        text = str(content or "")
        assert text, "LLM response content is empty"
        capability_indicators = [
            "help",
            "assist",
            "answer",
            "questions",
            "information",
            "support",
            "tasks",
            "conversation",
            "chat",
            "advice",
            "guidance",
            "explain",
            "provide",
            "offer",
            "capable",
            "able",
            "can",
            "do",
        ]
        found_capabilities = [indicator for indicator in capability_indicators if indicator in text.lower()]
        if not found_capabilities:
            pytest.fail(
                f"Expected response to contain capability indicators like 'help', 'assist', 'answer', etc. "
                f"Actual response: '{text[:100]}{'...' if len(text) > 100 else ''}'"
            )

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "enable_streaming",
        [
            pytest.param(False, id="streaming_disabled"),
        ],
    )
    def test_agents_rag_agent(
        self,
        unprivileged_llama_stack_client: LlamaStackClient,
        llama_stack_models: ModelInfo,
        vector_store_with_example_docs: VectorStore,
        enable_streaming: bool,
    ) -> None:
        """
        Test RAG agent that can answer questions about the Torchtune project using the documents
        in a vector database.

        Creates a RAG agent with PyTorch torchtune documentation, tests knowledge queries
        about fine-tuning techniques (LoRA, QAT, memory optimizations), and validates
        that responses contain expected technical keywords.

        Based on "Build a RAG Agent" example available at
        https://llamastack.github.io/docs/getting_started/detailed_tutorial

        Note: streaming is not tested (enable_streaming = False), as it seems to be broken in
        llama-stack 0.3.0 (Agents API is only partially implemented)
        """

        # Create the RAG agent connected to the vector database
        rag_agent = Agent(
            client=unprivileged_llama_stack_client,
            model=llama_stack_models.model_id,
            instructions="You are a helpful assistant. Use the file_search tool to answer questions as needed.",
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [vector_store_with_example_docs.id],
                }
            ],
        )
        session_id = rag_agent.create_session(session_name=f"s{uuid.uuid4().hex}")

        turns_with_expectations = get_torchtune_test_expectations()

        # Ask the agent about the inserted documents and validate responses
        validation_result = validate_rag_agent_responses(
            rag_agent=rag_agent,
            session_id=session_id,
            turns_with_expectations=turns_with_expectations,
            stream=enable_streaming,
            verbose=True,
            min_keywords_required=1,
            print_events=False,
        )

        # Assert that validation was successful
        assert validation_result["success"], (
            f"RAG agent validation failed with streaming={enable_streaming}. Summary: {validation_result['summary']}"
        )

        # Additional assertions for specific requirements
        for result in validation_result["results"]:
            assert result["response_length"] > 0, (
                f"No response content for question: {result['question']} (streaming={enable_streaming})"
            )
            assert len(result["found_keywords"]) > 0, (
                f"No expected keywords found in response for: {result['question']} (streaming={enable_streaming})"
            )
            if enable_streaming:
                assert result["event_count"] > 0, (
                    f"No events generated for question: {result['question']} (streaming={enable_streaming})"
                )
