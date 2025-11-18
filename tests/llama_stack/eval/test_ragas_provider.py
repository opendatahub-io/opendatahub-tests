from datetime import datetime

import pytest

from tests.llama_stack.constants import LlamaStackProviders
from tests.llama_stack.eval.utils import wait_for_eval_job_completion
from utilities.constants import MinIo, QWEN_MODEL_NAME

RAGAS_DATASET_ID: str = "ragas_dataset"
RAGAS_INLINE_BENCHMARK_ID = "ragas_benchmark_inline"


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection, llama_stack_server_config",
    [
        pytest.param(
            {"name": "test-llamastack-ragas"},
            MinIo.PodConfig.QWEN_HAP_BPIV2_MINIO_CONFIG,
            {"bucket": "llms"},
            {
                "vllm_url_fixture": "qwen_isvc_url",
                "inference_model": QWEN_MODEL_NAME,
                "embedding_model": "granite-embedding-125m",
            },
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
@pytest.mark.model_explainability
class TestLlamaStackRagasProvider:
    """Tests for LlamaStack Ragas evaluation provider integration."""

    def test_ragas_register_dataset(self, minio_pod, minio_data_connection, llama_stack_client):
        """Register a RAG evaluation dataset with sample question-answer data."""
        ragas_dataset = [
            {
                "user_input": "What is the capital of France?",
                "response": "The capital of France is Paris.",
                "retrieved_contexts": ["Paris is the capital and most populous city of France."],
                "reference": "Paris",
            },
        ]

        response = llama_stack_client.datasets.register(
            dataset_id=RAGAS_DATASET_ID,
            purpose="eval/question-answer",
            source={"type": "rows", "rows": ragas_dataset},
            metadata={
                "provider_id": "localfs",
                "description": "Sample RAG evaluation dataset for Ragas demo",
                "size": len(ragas_dataset),
                "format": "ragas",
                "created_at": datetime.now().isoformat(),
            },
        )

        assert response.identifier == RAGAS_DATASET_ID
        assert response.source.rows == ragas_dataset

    def test_ragas_register_benchmark(self, minio_pod, minio_data_connection, llama_stack_client):
        """Register a Ragas benchmark with answer relevancy scoring function."""
        llama_stack_client.benchmarks.register(
            benchmark_id=RAGAS_INLINE_BENCHMARK_ID,
            dataset_id=RAGAS_DATASET_ID,
            scoring_functions=["answer_relevancy"],
            provider_id=LlamaStackProviders.Eval.TRUSTYAI_RAGAS_INLINE,
        )

        response = llama_stack_client.benchmarks.list()
        assert response[0].dataset_id == RAGAS_DATASET_ID
        assert response[0].identifier == RAGAS_INLINE_BENCHMARK_ID
        assert response[0].provider_id == LlamaStackProviders.Eval.TRUSTYAI_RAGAS_INLINE

    def test_ragas_run_eval(self, minio_pod, minio_data_connection, llama_stack_client):
        """Run an evaluation job using the Ragas benchmark and wait for completion."""
        job = llama_stack_client.alpha.eval.run_eval(
            benchmark_id=RAGAS_INLINE_BENCHMARK_ID,
            benchmark_config={
                "eval_candidate": {
                    "model": QWEN_MODEL_NAME,
                    "type": "model",
                    "provider_id": LlamaStackProviders.Eval.TRUSTYAI_RAGAS_INLINE,
                    "sampling_params": {"temperature": 0.1, "max_tokens": 100},
                },
                "scoring_params": {},
            },
        )

        wait_for_eval_job_completion(
            llama_stack_client=llama_stack_client, job_id=job.job_id, benchmark_id=RAGAS_INLINE_BENCHMARK_ID
        )
