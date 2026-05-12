import os
from collections.abc import Generator
from typing import Any

import httpx
import pytest
import structlog
from llama_stack_client import LlamaStackClient
from llama_stack_client.types.vector_store import VectorStore
from ragas import SingleTurnSample

from tests.llama_stack.constants import (
    RAGAS_MAX_SAMPLES,
    ModelInfo,
)
from tests.llama_stack.datasets import Dataset
from tests.llama_stack.utils import extract_retrieved_contexts

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="class")
def ragas_evaluator_llm(
    unprivileged_llama_stack_client: LlamaStackClient,
    llama_stack_models: ModelInfo,
) -> Generator[Any, Any, Any]:
    """Create a RAGAS evaluator LLM backed by the Llama Stack OpenAI-compatible endpoint."""
    pytest.importorskip("openai")
    from openai import OpenAI
    from ragas.llms import llm_factory

    base_url = str(unprivileged_llama_stack_client.base_url).rstrip("/")
    verify_ssl = os.getenv("LLS_CLIENT_VERIFY_SSL", "false").lower() == "true"

    http_client = httpx.Client(verify=verify_ssl, timeout=httpx.Timeout(240.0))
    try:
        openai_client = OpenAI(
            api_key=os.getenv("LLS_CORE_VLLM_API_TOKEN", ""),
            base_url=f"{base_url}/v1",
            http_client=http_client,
        )

        evaluator_llm = llm_factory(
            model=llama_stack_models.model_id,
            provider="openai",
            client=openai_client,
        )
        evaluator_llm.model_args["max_tokens"] = 4096

        yield evaluator_llm
    finally:
        http_client.close()


class OpenAIEmbeddingsAdapter:
    """Adapter bridging ragas.embeddings.OpenAIEmbeddings to the embed_query/embed_documents
    interface that older ragas metrics (AnswerRelevancy) still expect internally."""

    def __init__(self, ragas_embeddings: Any):
        self._inner = ragas_embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_text(text=text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_texts(texts=texts)


@pytest.fixture(scope="class")
def ragas_evaluator_embeddings(
    unprivileged_llama_stack_client: LlamaStackClient,
    llama_stack_models: ModelInfo,
) -> Generator[Any, Any, Any]:
    """Create RAGAS embeddings backed by the Llama Stack vLLM embedding provider.

    Uses ragas.embeddings.OpenAIEmbeddings with a thin adapter that exposes
    embed_query()/embed_documents() for older metrics like AnswerRelevancy,
    avoiding the langchain-openai dependency.
    """
    from openai import OpenAI
    from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings

    base_url = str(unprivileged_llama_stack_client.base_url).rstrip("/")
    verify_ssl = os.getenv("LLS_CLIENT_VERIFY_SSL", "false").lower() == "true"

    http_client = httpx.Client(verify=verify_ssl, timeout=httpx.Timeout(120.0))
    try:
        openai_client = OpenAI(
            api_key=os.getenv("LLS_CORE_VLLM_API_TOKEN", "not-required"),
            base_url=f"{base_url}/v1",
            http_client=http_client,
        )

        ragas_embeddings = RagasOpenAIEmbeddings(
            client=openai_client,
            model=llama_stack_models.embedding_model.id,
        )

        yield OpenAIEmbeddingsAdapter(ragas_embeddings=ragas_embeddings)
    finally:
        http_client.close()


@pytest.fixture(scope="class")
def ragas_samples(
    unprivileged_llama_stack_client: LlamaStackClient,
    llama_stack_models: ModelInfo,
    vector_store: VectorStore,
    dataset: Dataset,
) -> list[SingleTurnSample]:
    """Build RAGAS evaluation samples by querying the RAG pipeline for each ground-truth QA pair.

    Uses the Responses API with the file_search tool against the vector store,
    mirroring a real-world RAG scenario.  The number of questions sent to the
    LLM is capped by ``RAGAS_MAX_SAMPLES`` (env var, default 5).
    """
    if RAGAS_MAX_SAMPLES < 1:
        raise pytest.UsageError("RAGAS_MAX_SAMPLES must be >= 1")

    qa_records = dataset.load_qa(retrieval_mode="vector")[:RAGAS_MAX_SAMPLES]
    if not qa_records:
        raise pytest.UsageError("No vector QA records available for RAGAS evaluation")

    samples: list[SingleTurnSample] = []

    for i, record in enumerate(qa_records):
        LOGGER.info(f"[{i + 1}/{len(qa_records)}] {record.question[:80]}...")

        try:
            resp = unprivileged_llama_stack_client.responses.create(
                model=llama_stack_models.model_id,
                instructions=(
                    "/no_think\n"
                    "You are a helpful assistant with access to data via the file_search tool.\n\n"
                    "When asked questions, use available tools to find the answer. Follow these rules:\n"
                    "1. Use tools immediately without asking for confirmation\n"
                    "2. Chain tool calls as needed\n"
                    "3. Do not narrate your process\n"
                    "4. Only provide the final answer\n"
                    "5. If the answer is not found in the context, respond with 'I don't know'"
                ),
                tools=[{"type": "file_search", "vector_store_ids": [vector_store.id]}],
                stream=False,
                input=record.question,
            )
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"RAG call failed for question {record.question!r}: {exc}")

        rag_answer = resp.output_text.strip()
        retrieved_contexts = extract_retrieved_contexts(response=resp)

        assert rag_answer, f"Empty RAG response for question: {record.question!r}"
        assert retrieved_contexts, f"No retrieved contexts for question: {record.question!r}"

        samples.append(
            SingleTurnSample(
                user_input=record.question,
                retrieved_contexts=retrieved_contexts,
                response=rag_answer,
                reference=record.ground_truth,
            )
        )

        LOGGER.info(f"  Answer: {rag_answer[:120]}...")
        LOGGER.info(f"  Retrieved {len(retrieved_contexts)} context(s)")

    assert len(samples) == len(qa_records), f"Built {len(samples)} RAGAS samples from {len(qa_records)} QA records"
    LOGGER.info(f"Built {len(samples)} RAGAS evaluation samples from {len(qa_records)} QA records")
    return samples
