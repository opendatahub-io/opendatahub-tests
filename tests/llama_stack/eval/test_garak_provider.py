from typing import Any

import pytest
import yaml
from llama_stack_client import BadRequestError
from simple_logger.logger import get_logger

from tests.llama_stack.constants import LlamaStackProviders
from tests.llama_stack.eval.constants import (
    GARAK_QUICK_BENCHMARK_ID,
    GARAK_CUSTOM_BENCHMARK_ID,
    GARAK_SHIELD_BENCHMARK_ID,
    GARAK_DEFAULT_TIMEOUT,
    GARAK_SHIELD_TIMEOUT,
    GARAK_SHIELD_ID,
    LLAMA_STACK_DISTRIBUTION_IMAGE,
    QWEN_LLAMA_STACK_MODEL_ID,
)
from tests.llama_stack.eval.utils import (
    wait_for_eval_job_completion,
    validate_eval_result_structure,
    validate_job_metadata,
)
from utilities.constants import QWEN_MODEL_NAME, MinIo, CHAT_GENERATION_CONFIG, BUILTIN_DETECTOR_CONFIG

LOGGER = get_logger(name=__name__)


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection, llama_stack_server_config",
    [
        pytest.param(
            {"name": "test-garak-quick"},
            MinIo.PodConfig.QWEN_HAP_BPIV2_MINIO_CONFIG,
            {"bucket": "llms"},
            {
                "vllm_url_fixture": "qwen_isvc_service_url",
                "inference_model": QWEN_MODEL_NAME,
                "enable_garak_remote": True,
                "distribution_image": LLAMA_STACK_DISTRIBUTION_IMAGE,
            },
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
@pytest.mark.model_explainability
@pytest.mark.llama_stack
@pytest.mark.smoke
class TestGarakRemoteQuickScan:
    """Tests for the pre-defined trustyai_garak::quick benchmark on Kubeflow Pipelines.

    Given a LlamaStack distribution with the Garak remote provider enabled,
    and a DSPA (DataSciencePipelinesApplication) deployed in the namespace,
    verify the full eval lifecycle: register, run, poll status, and retrieve results.
    """

    _job_id: str = ""

    @pytest.mark.dependency(name="quick_register")
    def test_register_quick_benchmark(
        self, minio_pod: Any, minio_data_connection: Any, llama_stack_client: Any
    ) -> None:
        """Given a LlamaStack client with the Garak remote provider,
        when registering the pre-defined quick benchmark (unregistering first if needed),
        then it appears in the benchmarks list with the correct provider.
        """
        quick_metadata = {
            "garak_config": {
                "run": {"soft_probe_prompt_cap": 10},
            },
        }
        try:
            llama_stack_client.alpha.benchmarks.register(
                benchmark_id=GARAK_QUICK_BENCHMARK_ID,
                dataset_id=GARAK_QUICK_BENCHMARK_ID,
                scoring_functions=["garak_scoring"],
                provider_id=LlamaStackProviders.Eval.TRUSTYAI_GARAK_REMOTE,
                provider_benchmark_id=GARAK_QUICK_BENCHMARK_ID,
                metadata=quick_metadata,
            )
        except BadRequestError as e:
            if "already exists" not in str(e):
                raise
            LOGGER.info(f"Benchmark {GARAK_QUICK_BENCHMARK_ID} already exists, unregistering and re-registering")
            llama_stack_client.alpha.benchmarks.unregister(benchmark_id=GARAK_QUICK_BENCHMARK_ID)
            llama_stack_client.alpha.benchmarks.register(
                benchmark_id=GARAK_QUICK_BENCHMARK_ID,
                dataset_id=GARAK_QUICK_BENCHMARK_ID,
                scoring_functions=["garak_scoring"],
                provider_id=LlamaStackProviders.Eval.TRUSTYAI_GARAK_REMOTE,
                provider_benchmark_id=GARAK_QUICK_BENCHMARK_ID,
                metadata=quick_metadata,
            )

        benchmarks = llama_stack_client.alpha.benchmarks.list()
        registered = [b for b in benchmarks if b.identifier == GARAK_QUICK_BENCHMARK_ID]
        assert len(registered) == 1, f"Expected 1 benchmark with id {GARAK_QUICK_BENCHMARK_ID}, got {len(registered)}"
        assert registered[0].provider_id == LlamaStackProviders.Eval.TRUSTYAI_GARAK_REMOTE

    @pytest.mark.dependency(name="quick_run", depends=["quick_register"])
    def test_run_quick_eval(self, minio_pod: Any, minio_data_connection: Any, llama_stack_client: Any) -> None:
        """Given a registered quick benchmark,
        when running an eval with the target model,
        then a Job is returned with a non-empty job_id.
        """
        job = llama_stack_client.alpha.eval.run_eval(
            benchmark_id=GARAK_QUICK_BENCHMARK_ID,
            benchmark_config={
                "eval_candidate": {
                    "type": "model",
                    "model": QWEN_LLAMA_STACK_MODEL_ID,
                    "sampling_params": {
                        "max_tokens": 100,
                    },
                }
            },
        )

        assert job.job_id, "run_eval must return a job with a non-empty job_id"
        TestGarakRemoteQuickScan._job_id = job.job_id
        LOGGER.info(f"Garak quick eval job started: {job.job_id}")

    @pytest.mark.dependency(name="quick_status", depends=["quick_run"])
    def test_quick_eval_status_and_completion(
        self, minio_pod: Any, minio_data_connection: Any, llama_stack_client: Any
    ) -> None:
        """Given a running quick eval job,
        when polling the job status,
        then it transitions to 'completed' and metadata contains kfp_run_id.
        """
        job_id = TestGarakRemoteQuickScan._job_id
        assert job_id, "No job_id from previous test"

        wait_for_eval_job_completion(
            llama_stack_client=llama_stack_client,
            job_id=job_id,
            benchmark_id=GARAK_QUICK_BENCHMARK_ID,
            wait_timeout=GARAK_DEFAULT_TIMEOUT,
        )

        completed_job = llama_stack_client.alpha.eval.jobs.status(
            job_id=job_id,
            benchmark_id=GARAK_QUICK_BENCHMARK_ID,
        )
        assert completed_job.status == "completed", f"Expected 'completed', got '{completed_job.status}'"
        validate_job_metadata(completed_job)

    @pytest.mark.dependency(name="quick_results", depends=["quick_status"])
    def test_quick_eval_results(self, minio_pod: Any, minio_data_connection: Any, llama_stack_client: Any) -> None:
        """Given a completed quick eval job,
        when retrieving the job result,
        then the EvaluateResponse contains generations, scores, and _overall metrics.
        """
        job_id = TestGarakRemoteQuickScan._job_id
        assert job_id, "No job_id from previous test"

        result = llama_stack_client.alpha.eval.jobs.retrieve(
            job_id=job_id,
            benchmark_id=GARAK_QUICK_BENCHMARK_ID,
        )

        aggregated = validate_eval_result_structure(result)
        LOGGER.info(
            f"Quick scan results: "
            f"total_attempts={aggregated['total_attempts']}, "
            f"vulnerable_responses={aggregated['vulnerable_responses']}, "
            f"attack_success_rate={aggregated['attack_success_rate']}"
        )


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection, llama_stack_server_config",
    [
        pytest.param(
            {"name": "test-garak-custom"},
            MinIo.PodConfig.QWEN_HAP_BPIV2_MINIO_CONFIG,
            {"bucket": "llms"},
            {
                "vllm_url_fixture": "qwen_isvc_service_url",
                "inference_model": QWEN_MODEL_NAME,
                "enable_garak_remote": True,
                "distribution_image": LLAMA_STACK_DISTRIBUTION_IMAGE,
            },
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
@pytest.mark.model_explainability
@pytest.mark.llama_stack
@pytest.mark.tier1
class TestGarakRemoteCustomBenchmark:
    """Tests for a custom Garak benchmark using explicit garak_config metadata.

    Given a LlamaStack distribution with the Garak remote provider,
    verify that a benchmark with custom probe configuration can be registered,
    executed, and returns valid vulnerability results.
    """

    _job_id: str = ""

    @pytest.mark.dependency(name="custom_register")
    def test_register_custom_benchmark(
        self, minio_pod: Any, minio_data_connection: Any, llama_stack_client: Any
    ) -> None:
        """Given a LlamaStack client with the Garak remote provider,
        when registering a benchmark with explicit garak_config metadata,
        then it appears in the benchmarks list with the correct metadata.
        """
        custom_metadata = {
            "garak_config": {
                "plugins": {"probe_spec": "promptinject.HijackHateHumans"},
                "run": {"generations": 1, "eval_threshold": 0.5, "soft_probe_prompt_cap": 10},
            },
            "timeout": GARAK_DEFAULT_TIMEOUT,
        }
        try:
            llama_stack_client.alpha.benchmarks.register(
                benchmark_id=GARAK_CUSTOM_BENCHMARK_ID,
                dataset_id=GARAK_CUSTOM_BENCHMARK_ID,
                scoring_functions=["garak_scoring"],
                provider_id=LlamaStackProviders.Eval.TRUSTYAI_GARAK_REMOTE,
                metadata=custom_metadata,
            )
        except BadRequestError as e:
            if "already exists" not in str(e):
                raise
            LOGGER.info(f"Benchmark {GARAK_CUSTOM_BENCHMARK_ID} already exists, unregistering and re-registering")
            llama_stack_client.alpha.benchmarks.unregister(benchmark_id=GARAK_CUSTOM_BENCHMARK_ID)
            llama_stack_client.alpha.benchmarks.register(
                benchmark_id=GARAK_CUSTOM_BENCHMARK_ID,
                dataset_id=GARAK_CUSTOM_BENCHMARK_ID,
                scoring_functions=["garak_scoring"],
                provider_id=LlamaStackProviders.Eval.TRUSTYAI_GARAK_REMOTE,
                metadata=custom_metadata,
            )

        benchmarks = llama_stack_client.alpha.benchmarks.list()
        registered = [b for b in benchmarks if b.identifier == GARAK_CUSTOM_BENCHMARK_ID]
        assert len(registered) == 1, f"Expected 1 benchmark with id {GARAK_CUSTOM_BENCHMARK_ID}, got {len(registered)}"
        assert registered[0].provider_id == LlamaStackProviders.Eval.TRUSTYAI_GARAK_REMOTE
        assert "garak_config" in registered[0].metadata, "Benchmark metadata must contain 'garak_config'"

    @pytest.mark.dependency(name="custom_run", depends=["custom_register"])
    def test_run_custom_eval(self, minio_pod: Any, minio_data_connection: Any, llama_stack_client: Any) -> None:
        """Given a registered custom benchmark,
        when running an eval with the target model,
        then the job completes successfully.
        """
        job = llama_stack_client.alpha.eval.run_eval(
            benchmark_id=GARAK_CUSTOM_BENCHMARK_ID,
            benchmark_config={
                "eval_candidate": {
                    "type": "model",
                    "model": QWEN_LLAMA_STACK_MODEL_ID,
                    "sampling_params": {
                        "max_tokens": 100,
                    },
                }
            },
        )

        assert job.job_id, "run_eval must return a job with a non-empty job_id"
        TestGarakRemoteCustomBenchmark._job_id = job.job_id
        LOGGER.info(f"Garak custom eval job started: {job.job_id}")

        wait_for_eval_job_completion(
            llama_stack_client=llama_stack_client,
            job_id=job.job_id,
            benchmark_id=GARAK_CUSTOM_BENCHMARK_ID,
            wait_timeout=GARAK_DEFAULT_TIMEOUT,
        )

    @pytest.mark.dependency(name="custom_results", depends=["custom_run"])
    def test_custom_eval_results(self, minio_pod: Any, minio_data_connection: Any, llama_stack_client: Any) -> None:
        """Given a completed custom eval job,
        when retrieving the job result,
        then the response contains generations and probe-level scores.
        """
        job_id = TestGarakRemoteCustomBenchmark._job_id
        assert job_id, "No job_id from previous test"

        result = llama_stack_client.alpha.eval.jobs.retrieve(
            job_id=job_id,
            benchmark_id=GARAK_CUSTOM_BENCHMARK_ID,
        )

        aggregated = validate_eval_result_structure(result)

        probe_scores = {k: v for k, v in result.scores.items() if k != "_overall"}
        assert len(probe_scores) > 0, "Expected at least one probe-level score besides _overall"
        LOGGER.info(f"Custom scan probes tested: {list(probe_scores.keys())}")
        LOGGER.info(
            f"Custom scan results: "
            f"total_attempts={aggregated['total_attempts']}, "
            f"attack_success_rate={aggregated['attack_success_rate']}"
        )


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection, "
    "orchestrator_config, guardrails_orchestrator, llama_stack_server_config",
    [
        pytest.param(
            {"name": "test-garak-shield"},
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
            {
                "orchestrator_config": True,
                "enable_built_in_detectors": True,
                "enable_guardrails_gateway": False,
            },
            {
                "vllm_url_fixture": "qwen_isvc_service_url",
                "inference_model": QWEN_MODEL_NAME,
                "fms_orchestrator_url_fixture": "guardrails_orchestrator_url",
                "enable_garak_remote": True,
                "distribution_image": LLAMA_STACK_DISTRIBUTION_IMAGE,
            },
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
@pytest.mark.usefixtures("patched_dsc_kserve_headed", "orchestrator_config", "guardrails_orchestrator")
@pytest.mark.model_explainability
@pytest.mark.llama_stack
@pytest.mark.tier2
class TestGarakRemoteShieldScan:
    """Tests for Garak shield scanning via the function-based generator path.

    Given a LlamaStack distribution with the Garak remote provider and
    FMS guardrails orchestrator deployed, verify that vulnerability scans
    run through input/output shields and produce valid results.
    """

    _job_id: str = ""

    @pytest.mark.dependency(name="shield_register_shield")
    def test_register_shield(
        self,
        current_client_token: str,
        minio_pod: Any,
        minio_data_connection: Any,
        llama_stack_client: Any,
    ) -> None:
        """Given a LlamaStack client with FMS guardrails provider,
        when registering a content_safety shield with regex PII detection,
        then the shield appears in the shields list.
        """
        shield_params = {
            "type": "content",
            "confidence_threshold": 0.5,
            "message_types": ["system", "user"],
            "auth_token": current_client_token,
            "verify_ssl": False,
            "detectors": {"regex": {"detector_params": {"regex": ["email", "ssn", "credit-card"]}}},
        }

        try:
            llama_stack_client.shields.register(
                shield_id=GARAK_SHIELD_ID,
                provider_shield_id=GARAK_SHIELD_ID,
                provider_id=LlamaStackProviders.Safety.TRUSTYAI_FMS,
                params=shield_params,
                timeout=120,
            )
        except BadRequestError as e:
            if "already exists" not in str(e):
                raise
            LOGGER.info(f"Shield {GARAK_SHIELD_ID} already exists, unregistering and re-registering")
            llama_stack_client.shields.unregister(shield_id=GARAK_SHIELD_ID)
            llama_stack_client.shields.register(
                shield_id=GARAK_SHIELD_ID,
                provider_shield_id=GARAK_SHIELD_ID,
                provider_id=LlamaStackProviders.Safety.TRUSTYAI_FMS,
                params=shield_params,
                timeout=120,
            )

        shields = llama_stack_client.shields.list()
        registered = [s for s in shields if s.identifier == GARAK_SHIELD_ID]
        assert len(registered) == 1, f"Expected 1 shield with id {GARAK_SHIELD_ID}, got {len(registered)}"
        assert registered[0].provider_id == LlamaStackProviders.Safety.TRUSTYAI_FMS

    @pytest.mark.dependency(name="shield_register_benchmark", depends=["shield_register_shield"])
    def test_register_shield_benchmark(
        self, minio_pod: Any, minio_data_connection: Any, llama_stack_client: Any
    ) -> None:
        """Given a registered content_safety shield,
        when registering a Garak benchmark with shield_ids,
        then the benchmark is created with shield configuration in metadata.
        """
        shield_metadata = {
            "garak_config": {
                "plugins": {"probe_spec": "promptinject.HijackHateHumans"},
                "run": {"generations": 1, "eval_threshold": 0.5, "soft_probe_prompt_cap": 10},
            },
            "shield_ids": [GARAK_SHIELD_ID],
            "timeout": GARAK_SHIELD_TIMEOUT,
        }
        try:
            llama_stack_client.alpha.benchmarks.register(
                benchmark_id=GARAK_SHIELD_BENCHMARK_ID,
                dataset_id=GARAK_SHIELD_BENCHMARK_ID,
                scoring_functions=["garak_scoring"],
                provider_id=LlamaStackProviders.Eval.TRUSTYAI_GARAK_REMOTE,
                metadata=shield_metadata,
            )
        except BadRequestError as e:
            if "already exists" not in str(e):
                raise
            LOGGER.info(f"Benchmark {GARAK_SHIELD_BENCHMARK_ID} already exists, unregistering and re-registering")
            llama_stack_client.alpha.benchmarks.unregister(benchmark_id=GARAK_SHIELD_BENCHMARK_ID)
            llama_stack_client.alpha.benchmarks.register(
                benchmark_id=GARAK_SHIELD_BENCHMARK_ID,
                dataset_id=GARAK_SHIELD_BENCHMARK_ID,
                scoring_functions=["garak_scoring"],
                provider_id=LlamaStackProviders.Eval.TRUSTYAI_GARAK_REMOTE,
                metadata=shield_metadata,
            )

        benchmarks = llama_stack_client.alpha.benchmarks.list()
        registered = [b for b in benchmarks if b.identifier == GARAK_SHIELD_BENCHMARK_ID]
        assert len(registered) == 1, f"Expected 1 benchmark with id {GARAK_SHIELD_BENCHMARK_ID}, got {len(registered)}"
        assert "shield_ids" in registered[0].metadata, "Benchmark metadata must contain 'shield_ids'"
        assert GARAK_SHIELD_ID in registered[0].metadata["shield_ids"]

    @pytest.mark.dependency(name="shield_run", depends=["shield_register_benchmark"])
    def test_run_shield_eval(self, minio_pod: Any, minio_data_connection: Any, llama_stack_client: Any) -> None:
        """Given a registered shield benchmark,
        when running an eval that routes through the SimpleShieldOrchestrator,
        then the job completes successfully.
        """
        job = llama_stack_client.alpha.eval.run_eval(
            benchmark_id=GARAK_SHIELD_BENCHMARK_ID,
            benchmark_config={
                "eval_candidate": {
                    "type": "model",
                    "model": QWEN_LLAMA_STACK_MODEL_ID,
                    "sampling_params": {
                        "max_tokens": 100,
                    },
                }
            },
        )

        assert job.job_id, "run_eval must return a job with a non-empty job_id"
        TestGarakRemoteShieldScan._job_id = job.job_id
        LOGGER.info(f"Garak shield eval job started: {job.job_id}")

        wait_for_eval_job_completion(
            llama_stack_client=llama_stack_client,
            job_id=job.job_id,
            benchmark_id=GARAK_SHIELD_BENCHMARK_ID,
            wait_timeout=GARAK_SHIELD_TIMEOUT,
        )

    @pytest.mark.dependency(name="shield_results", depends=["shield_run"])
    def test_shield_eval_results(self, minio_pod: Any, minio_data_connection: Any, llama_stack_client: Any) -> None:
        """Given a completed shield eval job,
        when retrieving the job result,
        then the response contains generations routed through shields and valid scores.
        """
        job_id = TestGarakRemoteShieldScan._job_id
        assert job_id, "No job_id from previous test"

        result = llama_stack_client.alpha.eval.jobs.retrieve(
            job_id=job_id,
            benchmark_id=GARAK_SHIELD_BENCHMARK_ID,
        )

        aggregated = validate_eval_result_structure(result)
        LOGGER.info(
            f"Shield scan results: "
            f"total_attempts={aggregated['total_attempts']}, "
            f"vulnerable_responses={aggregated['vulnerable_responses']}, "
            f"attack_success_rate={aggregated['attack_success_rate']}"
        )
