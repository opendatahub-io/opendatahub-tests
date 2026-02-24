import pytest
import yaml
from timeout_sampler import TimeoutSampler

from tests.llama_stack.constants import LlamaStackProviders
from tests.llama_stack.eval.utils import wait_for_eval_job_completion
from utilities.constants import (
    BUILTIN_DETECTOR_CONFIG,
    CHAT_GENERATION_CONFIG,
    MinIo,
    QWEN_MODEL_NAME,
)

GARAK_REMOTE_BENCHMARK_ID = "garak_benchmark_remote_quick"
GARAK_PROVIDER_BENCHMARK_ID = "trustyai_garak::quick"
GARAK_SHIELD_BENCHMARK_ID = "garak_benchmark_remote_with_shield"
GARAK_SHIELD_ID = "garak_shield_scan_input_guard"


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
    """Tests for LlamaStack Garak remote evaluation provider integration with Kubeflow Pipelines."""

    def test_garak_remote_run_and_retrieve_eval(self, minio_pod, minio_data_connection, llama_stack_client):
        """
        Given a remote Garak provider with Kubeflow enabled.
        When a quick Garak benchmark is registered and executed.
        Then the eval job completes and results are retrievable.
        """
        llama_stack_client.alpha.benchmarks.register(
            benchmark_id=GARAK_REMOTE_BENCHMARK_ID,
            dataset_id=GARAK_REMOTE_BENCHMARK_ID,
            scoring_functions=["garak_scoring"],
            provider_id=LlamaStackProviders.Eval.TRUSTYAI_GARAK_REMOTE,
            provider_benchmark_id=GARAK_PROVIDER_BENCHMARK_ID,
            metadata={"timeout": 600},
        )

        benchmarks = llama_stack_client.alpha.benchmarks.list()
        benchmark = next((entry for entry in benchmarks if entry.identifier == GARAK_REMOTE_BENCHMARK_ID), None)
        assert benchmark is not None
        assert benchmark.provider_id == LlamaStackProviders.Eval.TRUSTYAI_GARAK_REMOTE

        job = llama_stack_client.alpha.eval.run_eval(
            benchmark_id=GARAK_REMOTE_BENCHMARK_ID,
            benchmark_config={
                "eval_candidate": {
                    "type": "model",
                    "model": QWEN_MODEL_NAME,
                    "provider_id": LlamaStackProviders.Eval.TRUSTYAI_GARAK_REMOTE,
                    "sampling_params": {},
                },
                "scoring_params": {},
            },
        )

        wait_for_eval_job_completion(
            llama_stack_client=llama_stack_client,
            job_id=job.job_id,
            benchmark_id=GARAK_REMOTE_BENCHMARK_ID,
            wait_timeout=1200,
        )

        status = llama_stack_client.alpha.eval.jobs.status(
            job_id=job.job_id,
            benchmark_id=GARAK_REMOTE_BENCHMARK_ID,
        )
        assert status.status == "completed"

        def _retrieve_with_scores():
            result = llama_stack_client.alpha.eval.jobs.retrieve(
                job_id=job.job_id,
                benchmark_id=GARAK_REMOTE_BENCHMARK_ID,
            )
            return result if result.scores else None

        job_result = None
        for sample in TimeoutSampler(wait_timeout=180, sleep=15, func=_retrieve_with_scores):
            if sample:
                job_result = sample
                break

        assert job_result is not None
        assert isinstance(job_result.generations, list)
        assert isinstance(job_result.scores, dict)
        assert "_overall" in job_result.scores


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection, "
    "orchestrator_config, guardrails_orchestrator, llama_stack_server_config",
    [
        pytest.param(
            {"name": "test-garak-remote-shield-scan"},
            MinIo.PodConfig.QWEN_HAP_BPIV2_MINIO_CONFIG,
            {"bucket": "llms"},
            {
                "orchestrator_config_data": {
                    "config.yaml": yaml.dump({
                        "chat_generation": CHAT_GENERATION_CONFIG,
                        "detectors": BUILTIN_DETECTOR_CONFIG,
                    })
                },
            },
            {"orchestrator_config": True, "enable_built_in_detectors": True, "enable_guardrails_gateway": False},
            {
                "vllm_url_fixture": "qwen_isvc_url",
                "inference_model": QWEN_MODEL_NAME,
                "fms_orchestrator_url_fixture": "guardrails_orchestrator_url",
                "embedding_provider": "sentence-transformers",
                "enable_garak_remote": True,
            },
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
@pytest.mark.model_explainability
@pytest.mark.usefixtures("patched_dsc_kserve_headed", "orchestrator_config", "guardrails_orchestrator")
class TestLlamaStackGarakRemoteShieldScan:
    """Tests for Garak remote evaluation with shield scanning enabled."""

    def test_garak_remote_run_with_shield_scan(
        self,
        current_client_token,
        minio_pod,
        minio_data_connection,
        llama_stack_client,
    ):
        """
        Given a registered shield in the TrustyAI FMS provider.
        When Garak remote eval is run with shield_ids metadata.
        Then the job completes and returns retrievable scores.
        """
        shield_params = {
            "type": "content",
            "confidence_threshold": 0.5,
            "message_types": ["system", "user"],
            "auth_token": current_client_token,
            "verify_ssl": False,
            "detectors": {
                "regex": {
                    "detector_params": {
                        "regex": ["email", "ssn", "credit-card", "^hello$"],
                    }
                }
            },
        }

        llama_stack_client.shields.register(
            shield_id=GARAK_SHIELD_ID,
            provider_shield_id=GARAK_SHIELD_ID,
            provider_id=LlamaStackProviders.Safety.TRUSTYAI_FMS,
            params=shield_params,
            timeout=120,
        )
        shields = llama_stack_client.shields.list()
        shield = next((entry for entry in shields if entry.identifier == GARAK_SHIELD_ID), None)
        assert shield is not None
        assert shield.provider_id == LlamaStackProviders.Safety.TRUSTYAI_FMS

        llama_stack_client.alpha.benchmarks.register(
            benchmark_id=GARAK_SHIELD_BENCHMARK_ID,
            dataset_id=GARAK_SHIELD_BENCHMARK_ID,
            scoring_functions=["garak_scoring"],
            provider_id=LlamaStackProviders.Eval.TRUSTYAI_GARAK_REMOTE,
            provider_benchmark_id=GARAK_PROVIDER_BENCHMARK_ID,
            metadata={
                "garak_config": {
                    "plugins": {"probe_spec": ["promptinject.HijackHateHumans"]},
                    "run": {"generations": 1, "soft_probe_prompt_cap": 10},
                },
                "shield_ids": [GARAK_SHIELD_ID],
                "timeout": 900,
            },
        )

        benchmarks = llama_stack_client.alpha.benchmarks.list()
        benchmark = next((entry for entry in benchmarks if entry.identifier == GARAK_SHIELD_BENCHMARK_ID), None)
        assert benchmark is not None
        assert benchmark.provider_id == LlamaStackProviders.Eval.TRUSTYAI_GARAK_REMOTE

        job = llama_stack_client.alpha.eval.run_eval(
            benchmark_id=GARAK_SHIELD_BENCHMARK_ID,
            benchmark_config={
                "eval_candidate": {
                    "type": "model",
                    "model": QWEN_MODEL_NAME,
                    "provider_id": LlamaStackProviders.Eval.TRUSTYAI_GARAK_REMOTE,
                    "sampling_params": {},
                },
                "scoring_params": {},
            },
        )

        wait_for_eval_job_completion(
            llama_stack_client=llama_stack_client,
            job_id=job.job_id,
            benchmark_id=GARAK_SHIELD_BENCHMARK_ID,
            wait_timeout=1200,
        )

        status = llama_stack_client.alpha.eval.jobs.status(
            job_id=job.job_id,
            benchmark_id=GARAK_SHIELD_BENCHMARK_ID,
        )
        assert status.status == "completed"

        def _retrieve_with_scores():
            result = llama_stack_client.alpha.eval.jobs.retrieve(
                job_id=job.job_id,
                benchmark_id=GARAK_SHIELD_BENCHMARK_ID,
            )
            return result if result.scores else None

        job_result = None
        for sample in TimeoutSampler(wait_timeout=180, sleep=15, func=_retrieve_with_scores):
            if sample:
                job_result = sample
                break

        assert job_result is not None
        assert isinstance(job_result.generations, list)
        assert isinstance(job_result.scores, dict)
        assert "_overall" in job_result.scores
