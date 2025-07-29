import uuid
from typing import List

import pytest
from llama_stack_client import Agent, LlamaStackClient, RAGDocument
from llama_stack_client.types import EmbeddingsResponse, QueryChunksResponse
from llama_stack_client.types.vector_io_insert_params import Chunk
from ocp_resources.deployment import Deployment
from simple_logger.logger import get_logger

from utilities.rag_utils import TurnExpectation, validate_rag_agent_responses

LOGGER = get_logger(name=__name__)


class TestRag:
    """
    Test suite for LlamaStack RAG (Retrieval-Augmented Generation) functionality.

    Validates core RAG features including deployment, inference, agents,
    vector databases, and document retrieval with the Red Hat LlamaStack Distribution.
    """

    @pytest.mark.smoke
    def test_llama_stack_server(
        self, llama_stack_distribution_deployment: Deployment, rag_lls_client: LlamaStackClient
    ) -> None:
        """
        Test LlamaStack Server deployment and verify required models are available.

        Validates that the LlamaStack distribution is properly deployed with:
        - LLM model for text generation
        - Embedding model for document encoding
        - Proper embedding dimension configuration
        """
        llama_stack_distribution_deployment.wait_for_replicas()

        models = rag_lls_client.models.list()
        assert models is not None, "No models returned from LlamaStackClient"

        llm_model = next((m for m in models if m.api_model_type == "llm"), None)
        assert llm_model is not None, "No LLM model found in available models"
        model_id = llm_model.identifier
        assert model_id is not None, "No identifier set in LLM model"

        embedding_model = next((m for m in models if m.api_model_type == "embedding"), None)
        assert embedding_model is not None, "No embedding model found in available models"
        embedding_model_id = embedding_model.identifier
        assert embedding_model_id is not None, "No embedding model returned from LlamaStackClient"
        assert "embedding_dimension" in embedding_model.metadata, "embedding_dimension not found in model metadata"
        embedding_dimension = embedding_model.metadata["embedding_dimension"]
        assert embedding_dimension is not None, "No embedding_dimension set in embedding model"

    @pytest.mark.smoke
    def test_rag_chat_completions(self, rag_lls_client: LlamaStackClient) -> None:
        """
        Test basic chat completion inference through LlamaStack client.

        Validates that the server can perform text generation using the chat completions API
        and provides factually correct responses.

        Based on the example available at
        https://llama-stack.readthedocs.io/en/latest/getting_started/detailed_tutorial.html#step-4-run-the-demos
        """
        models = rag_lls_client.models.list()
        model_id = next(m for m in models if m.api_model_type == "llm").identifier

        response = rag_lls_client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
        )
        assert len(response.choices) > 0, "No response after basic inference on llama-stack server"

        # Check if response has the expected structure and content
        content = response.choices[0].message.content
        assert content is not None, "LLM response content is None"
        assert "Paris" in content, "The LLM didn't provide the expected answer to the prompt"

    @pytest.mark.smoke
    def test_rag_inference_embeddings(self, rag_lls_client: LlamaStackClient) -> None:
        """
        Test embedding model functionality and vector generation.

        Validates that the server can generate properly formatted embedding vectors
        for text input with correct dimensions as specified in model metadata.
        """
        models = rag_lls_client.models.list()
        embedding_model = next(m for m in models if m.api_model_type == "embedding")
        embedding_dimension = embedding_model.metadata["embedding_dimension"]

        embeddings_response = rag_lls_client.inference.embeddings(
            model_id=embedding_model.identifier,
            contents=["First chunk of text"],
            output_dimension=embedding_dimension,  # type: ignore
        )
        assert isinstance(embeddings_response, EmbeddingsResponse)
        assert len(embeddings_response.embeddings) == 1
        assert isinstance(embeddings_response.embeddings[0], list)
        assert isinstance(embeddings_response.embeddings[0][0], float)

    @pytest.mark.smoke
    def test_rag_vector_io_ingestion_retrieval(self, rag_lls_client: LlamaStackClient) -> None:
        """
        Validates basic vector_db API in llama-stack using milvus

        Tests registering, inserting and retrieving information from a milvus vector db database

        Based on the example available at
        https://llama-stack.readthedocs.io/en/latest/building_applications/rag.html
        """
        models = rag_lls_client.models.list()
        embedding_model = next(m for m in models if m.api_model_type == "embedding")
        embedding_dimension = embedding_model.metadata["embedding_dimension"]

        # Create a vector database instance
        vector_db_id = f"v{uuid.uuid4().hex}"

        try:
            rag_lls_client.vector_dbs.register(
                vector_db_id=vector_db_id,
                embedding_model=embedding_model.identifier,
                embedding_dimension=embedding_dimension,  # type: ignore
                provider_id="milvus",
            )

            # Calculate embeddings
            embeddings_response = rag_lls_client.inference.embeddings(
                model_id=embedding_model.identifier,
                contents=["First chunk of text"],
                output_dimension=embedding_dimension,  # type: ignore
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
            rag_lls_client.vector_io.insert(vector_db_id=vector_db_id, chunks=chunks_with_embeddings)

            # Query the vector db to find the chunk
            chunks_response = rag_lls_client.vector_io.query(
                vector_db_id=vector_db_id, query="What do you know about..."
            )
            assert isinstance(chunks_response, QueryChunksResponse)
            assert len(chunks_response.chunks) > 0
            assert chunks_response.chunks[0].metadata["document_id"] == "doc1"
            assert chunks_response.chunks[0].metadata["source"] == "precomputed"

        finally:
            # Cleanup: unregister the vector database to prevent resource leaks
            try:
                rag_lls_client.vector_dbs.unregister(vector_db_id)
            except Exception as e:
                LOGGER.warning(f"Failed to unregister vector database {vector_db_id}: {e}")

    @pytest.mark.smoke
    def test_rag_simple_agent(self, rag_lls_client: LlamaStackClient) -> None:
        """
        Test basic agent creation and conversation capabilities.

        Validates agent creation, session management, and turn-based interactions
        with both identity and capability questions.

        Based on the example available at
        https://llama-stack.readthedocs.io/en/latest/getting_started/detailed_tutorial.html#step-4-run-the-demos
        """
        models = rag_lls_client.models.list()
        model_id = next(m for m in models if m.api_model_type == "llm").identifier
        agent = Agent(client=rag_lls_client, model=model_id, instructions="You are a helpful assistant.")
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
        content = response.output_message.content
        assert content is not None, "LLM response content is None"
        assert "answers" in content, "The LLM didn't provide the expected answer to the prompt"

    @pytest.mark.smoke
    def test_rag_build_rag_agent(self, rag_lls_client: LlamaStackClient) -> None:
        """
        Test full RAG pipeline with vector database integration and knowledge retrieval.

        Creates a RAG agent with PyTorch torchtune documentation, tests knowledge queries
        about fine-tuning techniques (LoRA, QAT, memory optimizations), and validates
        that responses contain expected technical keywords.

        Based on the example available at
        https://llama-stack.readthedocs.io/en/latest/getting_started/detailed_tutorial.html#step-4-run-the-demos
        """
        models = rag_lls_client.models.list()
        model_id = next(m for m in models if m.api_model_type == "llm").identifier
        embedding_model = next(m for m in models if m.api_model_type == "embedding")

        embedding_dimension = embedding_model.metadata["embedding_dimension"]

        # Create a vector database instance
        vector_db_id = f"v{uuid.uuid4().hex}"

        rag_lls_client.vector_dbs.register(
            vector_db_id=vector_db_id,
            embedding_model=embedding_model.identifier,
            embedding_dimension=embedding_dimension,
            provider_id="milvus",
        )

        try:
            # Create the RAG agent connected to the vector database
            rag_agent = Agent(
                client=rag_lls_client,
                model=model_id,
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

            rag_lls_client.tool_runtime.rag_tool.insert(
                documents=documents,
                vector_db_id=vector_db_id,
                chunk_size_in_tokens=512,
            )

            turns_with_expectations: List[TurnExpectation] = [
                {
                    "question": "what is torchtune",
                    "expected_keywords": ["torchtune", "pytorch", "fine-tuning", "training", "model"],
                    "description": "Should provide information about torchtune framework",
                },
                {
                    "question": "What do you know about LoRA?",
                    "expected_keywords": [
                        "LoRA",
                        "parameter",
                        "efficient",
                        "fine-tuning",
                        "reduce",
                    ],
                    "description": "Should provide information about LoRA (Low Rank Adaptation)",
                },
                {
                    "question": "How can I optimize model training for quantization?",
                    "expected_keywords": [
                        "Quantization-Aware Training",
                        "QAT",
                        "training",
                        "fine-tuning",
                        "fake",
                        "quantized",
                    ],
                    "description": "Should provide information about QAT (Quantization-Aware Training)",
                },
                {
                    "question": "Are there any memory optimizations for LoRA?",
                    "expected_keywords": ["QLoRA", "fine-tuning", "4-bit"],
                    "description": "Should provide information about QLoRA",
                },
                {
                    "question": "tell me about dora",
                    "expected_keywords": ["dora", "parameter", "magnitude", "direction", "fine-tuning"],
                    "description": "Should provide information about DoRA (Weight-Decomposed Low-Rank Adaptation)",
                },
            ]

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
                rag_lls_client.vector_dbs.unregister(vector_db_id)
            except Exception as e:
                LOGGER.warning(f"Failed to unregister vector database {vector_db_id}: {e}")

    def test_rag_pdf(self, rag_lls_client: LlamaStackClient) -> None:
        """
        Test RAG functionality with PDF documents.

        Creates a RAG agent with Docling PDF documentation, tests knowledge queries
        about Docling features, AI models, output formats, and capabilities, and validates
        that responses contain expected technical keywords.
        """
        models = rag_lls_client.models.list()
        model_id = None
        embedding_model = None

        # Iterate through the list of models to find the first LLM model (for model_id)
        # and the first embedding model (for embedding_model). Stop searching once both are found.
        for m in models:
            if m.api_model_type == "llm" and model_id is None:
                model_id = m.identifier
            if m.api_model_type == "embedding" and embedding_model is None:
                embedding_model = m
            if model_id is not None and embedding_model is not None:
                break

        embedding_dimension = embedding_model.metadata["embedding_dimension"]
        
        # Create a vector database instance
        vector_db_id = f"v{uuid.uuid4().hex}"

        rag_lls_client.vector_dbs.register(
            vector_db_id=vector_db_id,
            embedding_model=embedding_model.identifier,
            embedding_dimension=embedding_dimension,
            provider_id="milvus",
        )

        try:
            # Create the RAG agent connected to the vector database
            rag_agent = Agent(
                client=rag_lls_client,
                model=model_id,
                instructions="You are a helpful assistant. Use the RAG tool to answer questions as needed.",
                tools=[
                    {
                        "name": "builtin::rag/knowledge_search",
                        "args": {"vector_db_ids": [vector_db_id]},
                    }
                ],
            )
            session_id = rag_agent.create_session(session_name=f"s{uuid.uuid4().hex}")

            # Insert PDF documents about Docling
            pdf_files_urls = [
                "https://arxiv.org/pdf/2408.09869"
            ]
            documents = [
                RAGDocument(
                    document_id=f"num-{i}", 
                    content=file_url, 
                    mime_type="application/pdf", 
                    metadata={}
                )
                for i, file_url in enumerate(pdf_files_urls)
            ]
            
            rag_lls_client.tool_runtime.rag_tool.insert(
                documents=documents,
                vector_db_id=vector_db_id,
                chunk_size_in_tokens=512,
            )

            turns_with_expectations: List[TurnExpectation] = [
                {
                    "question": "What is Docling?",
                    "expected_keywords": ["PDF", "conversion", "open-source", "MIT"],
                    "description": "Should provide information about Docling framework",
                },
                {
                    "question": "What AI models power Docling?",
                    "expected_keywords": ["DocLayNet", "TableFormer", "layout", "analysis", "table", "structure"],
                    "description": "Should provide information about Docling's AI models",
                },
                {
                    "question": "What output formats does Docling support for converted PDF documents?",
                    "expected_keywords": ["JSON", "Markdown"],
                    "description": "Should provide information about Docling's output formats",
                },
                {
                    "question": "Where can users find documentation and examples for Docling?",
                    "expected_keywords": ["GitHub", "repository", "documentation", "examples", "DS4SD"],
                    "description": "Should provide information about Docling documentation location",
                },
                {
                    "question": "What is the processing pipeline of Docling?",
                    "expected_keywords": ["PDF", "backend", "AI", "models", "post-processing"],
                    "description": "Should provide information about Docling's processing pipeline",
                },
                {
                    "question": "What are the two PDF backend choices available in Docling?",
                    "expected_keywords": ["qpdf", "pypdfium", "docling-parse"],
                    "description": "Should provide information about Docling's PDF backends",
                },
                {
                    "question": "What is TableFormer?",
                    "expected_keywords": ["vision-transformer", "table", "structure", "row", "column"],
                    "description": "Should provide information about TableFormer model",
                },
                {
                    "question": "What OCR library does Docling use in its initial release?",
                    "expected_keywords": ["EasyOCR"],
                    "description": "Should provide information about Docling's OCR library",
                },
                {
                    "question": "How can users extend Docling's capabilities?",
                    "expected_keywords": ["BaseModelPipeline", "sub-classing"],
                    "description": "Should provide information about extending Docling",
                },
                {
                    "question": "What are some of the downstream applications for Docling's output?",
                    "expected_keywords": ["search", "retrieval", "RAG", "classification", "knowledge", "extraction"],
                    "description": "Should provide information about Docling's applications",
                },
            ]

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
            assert validation_result["success"], f"RAG PDF agent validation failed. Summary: {validation_result['summary']}"

            # Additional assertions for specific requirements
            for result in validation_result["results"]:
                assert result["response_length"] > 0, f"No response content for question: {result['question']}"
                assert len(result["found_keywords"]) > 0, (
                    f"No expected keywords found in response for: {result['question']}"
                )

        finally:
            # Cleanup: unregister the vector database to prevent resource leaks
            try:
                rag_lls_client.vector_dbs.unregister(vector_db_id)
            except Exception as e:
                LOGGER.warning(f"Failed to unregister vector database {vector_db_id}: {e}")
