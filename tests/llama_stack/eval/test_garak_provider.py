import pytest

from tests.llama_stack.constants import LlamaStackProviders
from tests.llama_stack.eval.utils import wait_for_eval_job_completion
from utilities.constants import MinIo, QWEN_MODEL_NAME

GARAK_REMOTE_BENCHMARK_ID = "trustyai_garak::prompt_injection"


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection, llama_stack_server_config",
    [
        pytest.param(
            {"name": "test-garak-remote-security"},
            MinIo.PodConfig.QWEN_HAP_BPIV2_MINIO_CONFIG,
            {"bucket": "llms"},
            {
                "vllm_url_fixture": "qwen_isvc_url",
                "inference_model": QWEN_MODEL_NAME,
                "enable_garak_remote": True,  # Injects ENABLE_KUBEFLOW_GARAK=true
            },
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
@pytest.mark.model_explainability
class TestLlamaStackGarakRemoteProvider:
    """Tests for LlamaStack Garak remote evaluation provider without dataset registration."""

    def test_garak_remote_register_benchmark(self, llama_stack_client):
        """Register a Garak benchmark. dataset_id is set to benchmark_id as garak provides its own data."""
        llama_stack_client.alpha.benchmarks.register(
            benchmark_id=GARAK_REMOTE_BENCHMARK_ID,
            dataset_id=GARAK_REMOTE_BENCHMARK_ID,
            scoring_functions=["string"],
            provider_id=LlamaStackProviders.Eval.TRUSTYAI_GARAK_REMOTE,
            provider_benchmark_id="prompt_injection",
            metadata={
                "probes": ["promptinject.HijackHateHumans"],
                "timeout": 600,
            },
        )

        response = llama_stack_client.alpha.benchmarks.list()
        assert response[0].identifier == GARAK_REMOTE_BENCHMARK_ID
        assert response[0].provider_id == LlamaStackProviders.Eval.TRUSTYAI_GARAK_REMOTE

    def test_garak_remote_run_eval(self, minio_pod, minio_data_connection, llama_stack_client):
        """Run the security scan on Kubeflow and verify the vulnerability results."""

        job = llama_stack_client.alpha.eval.run_eval(
            benchmark_id=GARAK_REMOTE_BENCHMARK_ID,
            benchmark_config={
                "eval_candidate": {
                    "type": "model",
                    "model": QWEN_MODEL_NAME,
                    "sampling_params": {},
                }
            },
        )

        wait_for_eval_job_completion(
            llama_stack_client=llama_stack_client,
            job_id=job.job_id,
            benchmark_id=GARAK_REMOTE_BENCHMARK_ID,
        )
