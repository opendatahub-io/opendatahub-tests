from typing import Any

import pytest
import structlog
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from tests.llama_stack.constants import (
    ANSWER_RELEVANCY_THRESHOLD,
    CONTEXT_PRECISION_THRESHOLD,
    CONTEXT_RECALL_THRESHOLD,
    FAITHFULNESS_THRESHOLD,
)
from tests.llama_stack.datasets import (
    FINANCE_DATASET,
)
from tests.llama_stack.utils import (
    mean_ragas_score,
)

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llama_stack_server_config, vector_store, dataset",
    [
        pytest.param(
            {"name": "test-llamastack-ragas-eval", "randomize_name": True},
            {
                "llama_stack_storage_size": "2Gi",
                "vector_io_provider": "pgvector",
                "embedding_provider": "vllm-embedding",
                "files_provider": "s3",
            },
            {"vector_io_provider": "pgvector", "dataset": FINANCE_DATASET},
            FINANCE_DATASET,
            id="ragas-eval:pgvector, embedding:vllm-embedding, dataset:FINANCE_DATASET",
        ),
    ],
    indirect=True,
)
@pytest.mark.rag
class TestLlamaStackRagasEval:
    """
    Integration tests for evaluating the LlamaStack Retrieval-Augmented Generation (RAG)
    pipeline using RAGAS metrics. This class runs end-to-end evaluations over the Responses
    API with vector store-backed file search, computing faithfulness, answer relevancy,
    context precision, and context recall. It ensures that LlamaStack's RAG responses
    are factual, relevant, and grounded in the retrieved context, using multiple metrics
    and data samples. The test configures model and provider parameters via pytest,
    and leverages fixtures to build the RAGAS evaluation dataset.
    """

    @pytest.mark.parametrize(
        "metric_cls, metric_key, threshold, needs_embeddings",
        [
            pytest.param(
                Faithfulness,
                "faithfulness",
                FAITHFULNESS_THRESHOLD,
                False,
                id="faithfulness",
                marks=pytest.mark.tier1,
            ),
            pytest.param(
                AnswerRelevancy,
                "answer_relevancy",
                ANSWER_RELEVANCY_THRESHOLD,
                True,
                id="answer_relevancy",
                marks=pytest.mark.tier2,
            ),
            pytest.param(
                ContextPrecision,
                "context_precision",
                CONTEXT_PRECISION_THRESHOLD,
                False,
                id="context_precision",
                marks=pytest.mark.tier2,
            ),
            pytest.param(
                ContextRecall,
                "context_recall",
                CONTEXT_RECALL_THRESHOLD,
                False,
                id="context_recall",
                marks=pytest.mark.tier2,
            ),
        ],
    )
    def test_ragas_metric(
        self,
        ragas_samples: list[SingleTurnSample],
        ragas_evaluator_llm: Any,
        request: pytest.FixtureRequest,
        metric_cls: type,
        metric_key: str,
        threshold: float,
        needs_embeddings: bool,
    ) -> None:
        """Evaluate a RAGAS metric against RAG pipeline samples.

        Given: RAGAS samples from the RAG pipeline (Responses API + file_search)
        When: The specified metric is evaluated across all samples
        Then: The aggregate score meets the minimum threshold
        """
        kwargs: dict[str, Any] = {"llm": ragas_evaluator_llm}
        if needs_embeddings:
            kwargs["embeddings"] = request.getfixturevalue(argname="ragas_evaluator_embeddings")
        metric = metric_cls(**kwargs)

        result = evaluate(
            dataset=EvaluationDataset(samples=ragas_samples),
            metrics=[metric],
        )
        score = mean_ragas_score(scores=result[metric_key])
        LOGGER.info(f"RAGAS {metric_key} score: {score:.3f} (threshold: {threshold})")
        assert score >= threshold, f"RAGAS {metric_key} score {score:.3f} is below threshold {threshold}"
