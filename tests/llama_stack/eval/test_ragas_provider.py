from datetime import datetime

import pytest

from tests.llama_stack.constants import LlamaStackProviders
from tests.llama_stack.eval.utils import wait_for_eval_job_completion
from utilities.constants import QWEN_MODEL_NAME, MinIo

RAGAS_DATASET_ID: str = "ragas_dataset"
RAGAS_INLINE_BENCHMARK_ID = "ragas_benchmark_inline"
RAGAS_REMOTE_BENCHMARK_ID = "ragas_benchmark_remote"

RAGAS_TEST_DATASET = [
    {
        "user_input": "What is the capital of France?",
        "response": "The capital of France is Paris.",
        "retrieved_contexts": ["Paris is the capital and most populous city of France."],
        "reference": "Paris",
    },
]


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection, llama_stack_server_config",
    [
        pytest.param(
            {"name": "test-llamastack-ragas-inline"},
            MinIo.PodConfig.QWEN_HAP_BPIV2_MINIO_CONFIG,
            {"bucket": "llms"},
            {
                "vllm_url_fixture": "qwen_isvc_url",
                "inference_model": QWEN_MODEL_NAME,
                "embedding_provider": "sentence-transformers",
                "trustyai_embedding_model": "granite-embedding-125m-english",
            },
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
@pytest.mark.model_explainability
class TestLlamaStackRagasInlineProvider:
    """Tests for LlamaStack Ragas inline evaluation provider integration."""

    def test_ragas_inline_register_dataset(self, minio_pod, minio_data_connection, llama_stack_client):
        """Register a RAG evaluation dataset with sample question-answer data."""
        response = llama_stack_client.beta.datasets.register(
            dataset_id=RAGAS_DATASET_ID,
            purpose="eval/question-answer",
            source={"type": "rows", "rows": RAGAS_TEST_DATASET},
            metadata={
                "provider_id": "localfs",
                "description": "Sample RAG evaluation dataset for Ragas demo",
                "size": len(RAGAS_TEST_DATASET),
                "format": "ragas",
                "created_at": datetime.now().isoformat(),  # noqa: DTZ005
            },
        )

        assert response.identifier == RAGAS_DATASET_ID
        assert response.source.rows == RAGAS_TEST_DATASET

    def test_ragas_inline_register_benchmark(self, minio_pod, minio_data_connection, llama_stack_client):
        """Register a Ragas benchmark with answer relevancy scoring function."""
        llama_stack_client.alpha.benchmarks.register(
            benchmark_id=RAGAS_INLINE_BENCHMARK_ID,
            dataset_id=RAGAS_DATASET_ID,
            scoring_functions=["answer_relevancy"],
            provider_id=LlamaStackProviders.Eval.TRUSTYAI_RAGAS_INLINE,
        )

        response = llama_stack_client.alpha.benchmarks.list()
        assert response[0].dataset_id == RAGAS_DATASET_ID
        assert response[0].identifier == RAGAS_INLINE_BENCHMARK_ID
        assert response[0].provider_id == LlamaStackProviders.Eval.TRUSTYAI_RAGAS_INLINE

    def test_ragas_inline_run_eval(self, minio_pod, minio_data_connection, llama_stack_client):
        """Run an evaluation job using the Ragas inline benchmark and wait for completion."""
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
            llama_stack_client=llama_stack_client,
            job_id=job.job_id,
            benchmark_id=RAGAS_INLINE_BENCHMARK_ID,
        )


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection, llama_stack_server_config",
    [
        pytest.param(
            {"name": "test-llamastack-ragas-remote"},
            MinIo.PodConfig.QWEN_HAP_BPIV2_MINIO_CONFIG,
            {"bucket": "llms"},
            {
                "vllm_url_fixture": "qwen_isvc_url",
                "inference_model": QWEN_MODEL_NAME,
                "embedding_provider": "sentence-transformers",
                "trustyai_embedding_model": "granite-embedding-125m-english",
                "enable_ragas_remote": True,
            },
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
@pytest.mark.model_explainability
class TestLlamaStackRagasRemoteProvider:
    """Tests for LlamaStack Ragas remote evaluation provider integration with Kubeflow Pipelines."""

    def test_ragas_remote_register_dataset(self, minio_pod, minio_data_connection, llama_stack_client):
        """Register a RAG evaluation dataset with sample question-answer data."""
        response = llama_stack_client.beta.datasets.register(
            dataset_id=RAGAS_DATASET_ID,
            purpose="eval/question-answer",
            source={"type": "rows", "rows": RAGAS_TEST_DATASET},
            metadata={
                "provider_id": "localfs",
                "description": "Sample RAG evaluation dataset for Ragas demo",
                "size": len(RAGAS_TEST_DATASET),
                "format": "ragas",
                "created_at": datetime.now().isoformat(),  # noqa: DTZ005
            },
        )

        assert response.identifier == RAGAS_DATASET_ID
        assert response.source.rows == RAGAS_TEST_DATASET

    def test_ragas_remote_register_benchmark(self, minio_pod, minio_data_connection, llama_stack_client):
        """Register a Ragas benchmark with answer relevancy scoring function using remote provider."""
        llama_stack_client.alpha.benchmarks.register(
            benchmark_id=RAGAS_REMOTE_BENCHMARK_ID,
            dataset_id=RAGAS_DATASET_ID,
            scoring_functions=["answer_relevancy"],
            provider_id=LlamaStackProviders.Eval.TRUSTYAI_RAGAS_REMOTE,
        )

        response = llama_stack_client.alpha.benchmarks.list()
        assert response[0].dataset_id == RAGAS_DATASET_ID
        assert response[0].identifier == RAGAS_REMOTE_BENCHMARK_ID
        assert response[0].provider_id == LlamaStackProviders.Eval.TRUSTYAI_RAGAS_REMOTE

    def test_ragas_remote_run_eval(self, minio_pod, minio_data_connection, llama_stack_client):
        """Run an evaluation job using the Ragas remote benchmark and wait for completion."""
        job = llama_stack_client.alpha.eval.run_eval(
            benchmark_id=RAGAS_REMOTE_BENCHMARK_ID,
            benchmark_config={
                "eval_candidate": {
                    "model": QWEN_MODEL_NAME,
                    "type": "model",
                    "provider_id": LlamaStackProviders.Eval.TRUSTYAI_RAGAS_REMOTE,
                    "sampling_params": {"temperature": 0.1, "max_tokens": 100},
                },
                "scoring_params": {},
            },
        )

        wait_for_eval_job_completion(
            llama_stack_client=llama_stack_client,
            job_id=job.job_id,
            benchmark_id=RAGAS_REMOTE_BENCHMARK_ID,
        )
