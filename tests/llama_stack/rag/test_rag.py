import uuid
from typing import List

import pytest
from llama_stack_client import Agent, LlamaStackClient, RAGDocument
from llama_stack_client.types import EmbeddingsResponse, QueryChunksResponse
from llama_stack_client.types.vector_io_insert_params import Chunk
from simple_logger.logger import get_logger
from utilities.rag_utils import TurnExpectation, validate_rag_agent_responses, validate_api_responses

from tests.llama_stack.constants import TORCHTUNE_TEST_EXPECTATIONS

LOGGER = get_logger(name=__name__)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-llamastack-rag"},
        )
    ],
    indirect=True,
)
class TestLlamaStackRag:
    """
    Test suite for LlamaStack RAG (Retrieval-Augmented Generation) functionality.

    Validates core RAG features including deployment, inference, agents,
    vector databases, and document retrieval with the Red Hat LlamaStack Distribution.
    """

    def _get_torchtune_test_expectations(self) -> List[TurnExpectation]:
        """
        Helper method to get the test expectations for TorchTune documentation questions.

        Returns:
            List of TurnExpectation objects for testing RAG responses
        """
        return [
            {
                "question": expectation.question,
                "expected_keywords": expectation.expected_keywords,
                "description": expectation.description,
            }
            for expectation in TORCHTUNE_TEST_EXPECTATIONS
        ]

    def _create_response_function(
        self, llama_stack_client: LlamaStackClient, llama_stack_models, vector_store
    ) -> callable:
        """
        Helper method to create a response function for testing with vector store integration.

        Args:
            llama_stack_client: The LlamaStack client instance
            llama_stack_models: The model configuration
            vector_store: The vector store instance

        Returns:
            A callable function that takes a question and returns a response
        """

        def _response_fn(*, question: str) -> str:
            response = llama_stack_client.responses.create(
                input=question,
                model=llama_stack_models.model_id,
                tools=[
                    {
                        "type": "file_search",
                        "vector_store_ids": [vector_store.id],
                    }
                ],
            )
            return response.output_text

        return _response_fn

    @pytest.mark.smoke
    def test_rag_inference_embeddings(self, llama_stack_client: LlamaStackClient, llama_stack_models) -> None:
        """
        Test embedding model functionality and vector generation.

        Validates that the server can generate properly formatted embedding vectors
        for text input with correct dimensions as specified in model metadata.
        """
        embeddings_response = llama_stack_client.inference.embeddings(
            model_id=llama_stack_models.embedding_model.identifier,
            contents=["First chunk of text"],
            output_dimension=llama_stack_models.embedding_dimension,  # type: ignore
        )
        assert isinstance(embeddings_response, EmbeddingsResponse)
        assert len(embeddings_response.embeddings) == 1
        assert isinstance(embeddings_response.embeddings[0], list)
        assert isinstance(embeddings_response.embeddings[0][0], float)

    @pytest.mark.smoke
    def test_rag_vector_io_ingestion_retrieval(self, llama_stack_client: LlamaStackClient, llama_stack_models) -> None:
        """
        Validates basic vector_db API in llama-stack using milvus

        Tests registering, inserting and retrieving information from a milvus vector db database

        Based on the example available at
        https://llama-stack.readthedocs.io/en/latest/building_applications/rag.html
        """
        # Create a vector database instance
        vector_db_id = f"v{uuid.uuid4().hex}"

        try:
            llama_stack_client.vector_dbs.register(
                vector_db_id=vector_db_id,
                embedding_model=llama_stack_models.embedding_model.identifier,
                embedding_dimension=llama_stack_models.embedding_dimension,  # type: ignore
                provider_id="milvus",
            )

            # Calculate embeddings
            embeddings_response = llama_stack_client.inference.embeddings(
                model_id=llama_stack_models.embedding_model.identifier,
                contents=["First chunk of text"],
                output_dimension=llama_stack_models.embedding_dimension,  # type: ignore
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
            llama_stack_client.vector_io.insert(vector_db_id=vector_db_id, chunks=chunks_with_embeddings)

            # Query the vector db to find the chunk
            chunks_response = llama_stack_client.vector_io.query(
                vector_db_id=vector_db_id, query="What do you know about..."
            )
            assert isinstance(chunks_response, QueryChunksResponse)
            assert len(chunks_response.chunks) > 0
            assert chunks_response.chunks[0].metadata["document_id"] == "doc1"
            assert chunks_response.chunks[0].metadata["source"] == "precomputed"

        finally:
            # Cleanup: unregister the vector database to prevent resource leaks
            try:
                llama_stack_client.vector_dbs.unregister(vector_db_id)
            except Exception as e:
                LOGGER.warning(f"Failed to unregister vector database {vector_db_id}: {e}")

    @pytest.mark.smoke
    def test_rag_simple_agent(self, llama_stack_client: LlamaStackClient, llama_stack_models) -> None:
        """
        Test basic agent creation and conversation capabilities.

        Validates agent creation, session management, and turn-based interactions
        with both identity and capability questions.

        Based on the example available at
        https://llama-stack.readthedocs.io/en/latest/getting_started/detailed_tutorial.html#step-4-run-the-demos
        """
        agent = Agent(
            client=llama_stack_client, model=llama_stack_models.model_id, instructions="You are a helpful assistant."
        )
        s_id = agent.create_session(session_name=f"s{uuid.uuid4().hex}")

        # Test identity question
        response = agent.create_turn(
            messages=[{"role": "user", "content": "Who are you?"}],
            session_id=s_id,
            stream=False,
        )
        content = response.output_message.content
        assert content is not None, "LLM response content is None"
        assert "model" in content, "The LLM didn't provide the expected answer to the prompt"

        # Test capability question
        response = agent.create_turn(
            messages=[{"role": "user", "content": "What can you do?"}],
            session_id=s_id,
            stream=False,
        )
        content = response.output_message.content.lower()
        assert content is not None, "LLM response content is None"
        assert "answer" in content, "The LLM didn't provide the expected answer to the prompt"
        assert "translate" in content, "The LLM didn't provide the expected answer to the prompt"
        assert "summarize" in content, "The LLM didn't provide the expected answer to the prompt"
        assert "chat" in content, "The LLM didn't provide the expected answer to the prompt"

    @pytest.mark.smoke
    def test_rag_build_rag_agent(self, llama_stack_client: LlamaStackClient, llama_stack_models) -> None:
        """
        Test full RAG pipeline with vector database integration and knowledge retrieval.

        Creates a RAG agent with PyTorch torchtune documentation, tests knowledge queries
        about fine-tuning techniques (LoRA, QAT, memory optimizations), and validates
        that responses contain expected technical keywords.

        Based on the example available at
        https://llama-stack.readthedocs.io/en/latest/getting_started/detailed_tutorial.html#step-4-run-the-demos
        """
        # Create a vector database instance
        vector_db_id = f"v{uuid.uuid4().hex}"

        llama_stack_client.vector_dbs.register(
            vector_db_id=vector_db_id,
            embedding_model=llama_stack_models.embedding_model.identifier,
            embedding_dimension=llama_stack_models.embedding_dimension,
            provider_id="milvus",
        )

        try:
            # Create the RAG agent connected to the vector database
            rag_agent = Agent(
                client=llama_stack_client,
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

            llama_stack_client.tool_runtime.rag_tool.insert(
                documents=documents,
                vector_db_id=vector_db_id,
                chunk_size_in_tokens=512,
            )

            turns_with_expectations = self._get_torchtune_test_expectations()

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
            try:
                llama_stack_client.vector_dbs.unregister(vector_db_id)
            except Exception as e:
                LOGGER.warning(f"Failed to unregister vector database {vector_db_id}: {e}")

    @pytest.mark.smoke
    def test_rag_simple_responses(self, llama_stack_client: LlamaStackClient, llama_stack_models) -> None:
        """
        Test simple responses API from the llama-stack server.

        Validates basic text generation capabilities using the responses API endpoint.
        Tests identity and capability questions to ensure the LLM can provide
        appropriate responses about itself and its functionality.
        """

        response = llama_stack_client.responses.create(
            model=llama_stack_models.model_id,
            input="Who are you?",
            instructions="You are a helpful assistant.",
        )

        content = response.output_text
        assert content is not None, "LLM response content is None"
        assert any(
            answer in content.lower() for answer in ["model", "assistant", "ai", "artificial", "language model"]
        ), (
            f"The LLM didn't provide any of the expected answers "
            f"['model', 'assistant', 'ai', 'llm', 'language model']. Got: {content}"
        )

        response = llama_stack_client.responses.create(
            model=llama_stack_models.model_id,
            input="What can you do?",
            instructions="You are a helpful assistant.",
        )
        content = response.output_text
        assert content is not None, "LLM response content is None"
        assert "answer" in content, "The LLM didn't provide the expected answer to the prompt"

    @pytest.mark.smoke
    def test_rag_full_responses(
        self,
        llama_stack_client: LlamaStackClient,
        llama_stack_models,
        vector_store_with_docs,
    ) -> None:
        """
        Test responses API from the llama-stack server with vector store integration.

        Uses a vector store with pre-uploaded TorchTune documentation files and tests the responses API
        with file search capabilities. Validates that the API can retrieve and use
        knowledge from uploaded documents to answer questions.
        """

        _response_fn = self._create_response_function(
            llama_stack_client=llama_stack_client,
            llama_stack_models=llama_stack_models,
            vector_store=vector_store_with_docs,
        )

        turns_with_expectations = self._get_torchtune_test_expectations()

        validation_result = validate_api_responses(response_fn=_response_fn, test_cases=turns_with_expectations)

        assert validation_result["success"], f"RAG agent validation failed. Summary: {validation_result['summary']}"

    @pytest.mark.smoke
    def test_rag_vector_store_search(self, llama_stack_client: LlamaStackClient, vector_store_with_docs) -> None:
        """
        Test vector store search functionality using the search endpoint.

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
            search_response = llama_stack_client.vector_stores.search(
                vector_store_id=vector_store_with_docs.id, query=query
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
