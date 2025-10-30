import pytest

from tests.llama_stack.constants import LlamaStackProviders
from utilities.constants import Timeout, MinIo, QWEN_MODEL_NAME
from timeout_sampler import TimeoutSampler


TRUSTYAI_LMEVAL_ARCEASY = f"{LlamaStackProviders.Eval.TRUSTYAI_LMEVAL}::arc_easy"


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

        llama_stack_client.benchmarks.register(
            benchmark_id=TRUSTYAI_LMEVAL_ARCEASY,
            dataset_id=TRUSTYAI_LMEVAL_ARCEASY,
            scoring_functions=["string"],
            provider_id=LlamaStackProviders.Eval.TRUSTYAI_LMEVAL,
            provider_benchmark_id="string",
            metadata={"tokenized_requests": False, "tokenizer": "google/flan-t5-small"},
        )
        benchmarks = llama_stack_client.benchmarks.list()

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

        samples = TimeoutSampler(
            wait_timeout=Timeout.TIMEOUT_10MIN,
            sleep=30,
            func=lambda: llama_stack_client.alpha.eval.jobs.status(
                job_id=job.job_id, benchmark_id=TRUSTYAI_LMEVAL_ARCEASY
            ).status,
        )

        for sample in samples:
            if sample == "completed":
                break
