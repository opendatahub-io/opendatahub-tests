import pytest

from tests.llama_stack.constants import LlamaStackProviders
from tests.llama_stack.eval.utils import wait_for_eval_job_completion
from utilities.constants import MinIo, QWEN_MODEL_NAME


TRUSTYAI_LMEVAL_ARCEASY = f"{LlamaStackProviders.Eval.TRUSTYAI_LMEVAL}::arc_easy"
TRUSTYAI_LMEVAL_CUSTOM = f"{LlamaStackProviders.Eval.TRUSTYAI_LMEVAL}::dk-bench"


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection, llama_stack_server_config",
    [
        pytest.param(
            {"name": "test-llamastack-lmeval"},
            MinIo.PodConfig.QWEN_HAP_BPIV2_MINIO_CONFIG,
            {"bucket": "llms"},
            {
                "vllm_url_fixture": "qwen_isvc_url",
                "inference_model": QWEN_MODEL_NAME,
                "embedding_provider": "sentence-transformers",
            },
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
@pytest.mark.model_explainability
class TestLlamaStackLMEvalProvider:
    """
    Tests for the LlamaStack LMEval provider.
    1. Register the LLM that will be evaluated.
    2. Register the arc_easy benchmark.
    3. Run the evaluation and wait until it's completed.
    """

    def test_lmeval_register_benchmark(self, minio_pod, minio_data_connection, llama_stack_client):
        llama_stack_client.models.register(
            provider_id=LlamaStackProviders.Inference.VLLM_INFERENCE, model_type="llm", model_id=QWEN_MODEL_NAME
        )

        llama_stack_client.alpha.benchmarks.register(
            benchmark_id=TRUSTYAI_LMEVAL_ARCEASY,
            dataset_id=TRUSTYAI_LMEVAL_ARCEASY,
            scoring_functions=["string"],
            provider_id=LlamaStackProviders.Eval.TRUSTYAI_LMEVAL,
            provider_benchmark_id="string",
            metadata={"tokenized_requests": False, "tokenizer": "google/flan-t5-small"},
        )
        benchmarks = llama_stack_client.alpha.benchmarks.list()

        assert len(benchmarks) == 1
        assert benchmarks[0].identifier == TRUSTYAI_LMEVAL_ARCEASY
        assert benchmarks[0].provider_id == LlamaStackProviders.Eval.TRUSTYAI_LMEVAL

    def test_llamastack_run_eval(
        self, minio_pod, minio_data_connection, patched_dsc_lmeval_allow_all, llama_stack_client
    ):
        job = llama_stack_client.alpha.eval.run_eval(
            benchmark_id=TRUSTYAI_LMEVAL_ARCEASY,
            benchmark_config={
                "eval_candidate": {
                    "model": QWEN_MODEL_NAME,
                    "type": "model",
                    "provider_id": LlamaStackProviders.Eval.TRUSTYAI_LMEVAL,
                    "sampling_params": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 10},
                },
                "scoring_params": {},
                "num_examples": 1,
            },
        )

        wait_for_eval_job_completion(
            llama_stack_client=llama_stack_client, job_id=job.job_id, benchmark_id=TRUSTYAI_LMEVAL_ARCEASY
        )


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection, llama_stack_server_config",
    [
        pytest.param(
            {"name": "test-llamastack-lmeval-custom"},
            MinIo.PodConfig.QWEN_HAP_BPIV2_MINIO_CONFIG,
            {"bucket": "llms"},
            {
                "vllm_url_fixture": "qwen_isvc_url",
                "inference_model": QWEN_MODEL_NAME,
                "embedding_provider": "sentence-transformers",
            },
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
@pytest.mark.model_explainability
class TestLlamaStackLMEvalCustomBenchmark:
    """
    Tests for running LM-Eval on a custom benchmark (DK-Bench)
    using a preloaded dataset in a PVC.
    """

    def test_lmeval_register_custom_benchmark(
        self, minio_pod, minio_data_connection, dataset_pvc, dataset_upload, llama_stack_client, qwen_isvc_url
    ):
        llama_stack_client.models.register(
            provider_id=LlamaStackProviders.Inference.VLLM_INFERENCE, model_type="llm", model_id=QWEN_MODEL_NAME
        )

        llama_stack_client.alpha.benchmarks.register(
            benchmark_id=TRUSTYAI_LMEVAL_CUSTOM,
            dataset_id=TRUSTYAI_LMEVAL_CUSTOM,
            scoring_functions=["string"],
            provider_id=LlamaStackProviders.Eval.TRUSTYAI_LMEVAL,
            provider_benchmark_id="string",
            metadata={
                "custom_task": {
                    "git": {
                        "url": "https://github.com/trustyai-explainability/lm-eval-tasks.git",
                        "branch": "main",
                        "commit": "8220e2d73c187471acbe71659c98bccecfe77958",  # pragma: allowlist secret
                        "path": "tasks/",
                    }
                },
                "env": {
                    "DK_BENCH_DATASET_PATH": dataset_upload["dataset_path"],
                    "JUDGE_MODEL_URL": f"{qwen_isvc_url}/chat/completions",
                    "JUDGE_MODEL_NAME": QWEN_MODEL_NAME,
                    "JUDGE_API_KEY": "",
                },
                "tokenized_requests": False,
                "tokenizer": "google/flan-t5-small",
                "input": {"storage": {"pvc": "dataset-pvc"}},
            },
        )

        benchmarks = llama_stack_client.alpha.benchmarks.list()
        assert len(benchmarks) == 1
        assert benchmarks[0].identifier == TRUSTYAI_LMEVAL_CUSTOM
        assert benchmarks[0].provider_id == LlamaStackProviders.Eval.TRUSTYAI_LMEVAL

    def test_llamastack_run_custom_eval(
        self,
        minio_pod,
        minio_data_connection,
        dataset_pvc,
        dataset_upload,
        patched_dsc_lmeval_allow_all,
        llama_stack_client,
        teardown_lmeval_job_pod,
    ):
        job = llama_stack_client.alpha.eval.run_eval(
            benchmark_id=TRUSTYAI_LMEVAL_CUSTOM,
            benchmark_config={
                "eval_candidate": {
                    "model": QWEN_MODEL_NAME,
                    "type": "model",
                    "provider_id": LlamaStackProviders.Eval.TRUSTYAI_LMEVAL,
                    "sampling_params": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 10},
                },
                "num_examples": 2,
            },
        )

        wait_for_eval_job_completion(
            llama_stack_client=llama_stack_client, job_id=job.job_id, benchmark_id=TRUSTYAI_LMEVAL_CUSTOM
        )
