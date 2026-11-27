"""CLEAR direct MLflow trace ingestion tests (RHOAIENG-63558).

Exercises eval-hub-sdk#129 (TracesNamespace materialize) and eval-hub#585
(sidecar preserves /mlflow path prefix) via ibm-clear jobs.

Given: EvalHub + MLflow (/mlflow prefix) + seeded agent traces + CLEAR provider
When:  A CLEAR job is submitted with parameters.mlflow_traces_experiment_name
Then:  The job completes (or fails clearly for negative cases) with expected signals
"""

from __future__ import annotations

import pytest
import requests
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.service import Service

from tests.ai_safety.evalhub.clear.constants import (
    CLEAR_JOB_POLL_INTERVAL,
    CLEAR_JOB_TIMEOUT,
    CLEAR_RESULTS_EXPERIMENT_NAME,
    CLEAR_UNKNOWN_EXPERIMENT_NAME,
)
from tests.ai_safety.evalhub.clear.utils import (
    build_clear_job_payload,
    validate_clear_job_completed,
)
from tests.ai_safety.evalhub.constants import EVALHUB_JOBS_PATH
from tests.ai_safety.evalhub.utils import (
    EVALHUB_JOB_TERMINAL_STATES,
    build_headers,
    submit_evalhub_job,
    wait_for_evalhub_job,
)
from utilities.constants import Timeout


@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-evalhub-clear-mlflow"})],
    indirect=True,
)
@pytest.mark.ai_safety
@pytest.mark.usefixtures("evalhub_mt_ready")
class TestEvalHubClearMlflowTraces:
    """C-02/C-03/C-04 happy path and C-11/C-12 negatives for MLflow trace ingestion."""

    @pytest.mark.tier1
    @pytest.mark.slow
    def test_clear_job_ingests_mlflow_traces_and_completes(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
        clear_provider_id: str,
        clear_seeded_mlflow_traces_experiment: dict[str, str],
    ) -> None:
        """Given seeded MLflow agent traces, when CLEAR runs with mlflow_traces_experiment_name,
        then the job completes with CLEAR metrics (exercises sidecar /mlflow prefix)."""
        traces_experiment = clear_seeded_mlflow_traces_experiment["experiment_name"]
        payload = build_clear_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            provider_id=clear_provider_id,
            job_name="clear-mlflow-traces-e2e",
            experiment_name=CLEAR_RESULTS_EXPERIMENT_NAME,
            parameters={
                "mlflow_traces_experiment_name": traces_experiment,
                "mlflow_traces_max_results": 10,
            },
        )
        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        job_id = data["resource"]["id"]
        assert data.get("resource", {}).get("mlflow_experiment_id"), (
            f"Expected mlflow_experiment_id for results experiment on submit, got: {data.get('resource')}"
        )

        job_result = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
            timeout=CLEAR_JOB_TIMEOUT,
            sleep=CLEAR_JOB_POLL_INTERVAL,
        )
        validate_clear_job_completed(job_data=job_result, provider_id=clear_provider_id)

        benchmarks = job_result.get("results", {}).get("benchmarks", [])
        clear_bench = next(benchmark for benchmark in benchmarks if benchmark.get("id") == "agentic-evaluation")
        # Result upload is best-effort depending on adapter/MLflow; surface run id when present.
        if clear_bench.get("mlflow_run_id"):
            assert isinstance(clear_bench["mlflow_run_id"], str) and clear_bench["mlflow_run_id"]

    @pytest.mark.tier1
    def test_clear_job_submit_with_mlflow_traces_params_accepted(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
        clear_provider_id: str,
        clear_seeded_mlflow_traces_experiment: dict[str, str],
    ) -> None:
        """Given valid mlflow_traces_* parameters, when the job is submitted, then API returns 202."""
        payload = build_clear_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            provider_id=clear_provider_id,
            job_name="clear-mlflow-traces-submit",
            parameters={
                "mlflow_traces_experiment_name": clear_seeded_mlflow_traces_experiment["experiment_name"],
                "mlflow_traces_experiment_id": clear_seeded_mlflow_traces_experiment["experiment_id"],
                "mlflow_traces_max_results": 5,
            },
        )
        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        assert data.get("resource", {}).get("id"), f"Expected job id in response, got: {data}"

    @pytest.mark.tier2
    def test_clear_job_missing_trace_source_fails(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
        clear_provider_id: str,
    ) -> None:
        """Given no mlflow_traces_* and no staged test_data_ref, when CLEAR runs, then the job fails."""
        payload = build_clear_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            provider_id=clear_provider_id,
            job_name="clear-missing-trace-source",
            parameters={},
        )
        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        job_id = data["resource"]["id"]
        job_result = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
            timeout=Timeout.TIMEOUT_10MIN,
            sleep=CLEAR_JOB_POLL_INTERVAL,
        )
        state = job_result.get("status", {}).get("state")
        assert state in EVALHUB_JOB_TERMINAL_STATES - {"completed"}, (
            f"Expected non-completed terminal state for missing trace source, got '{state}': {job_result.get('status')}"
        )

    @pytest.mark.tier2
    def test_clear_job_unknown_mlflow_traces_experiment_fails(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
        clear_provider_id: str,
    ) -> None:
        """Given an unknown mlflow_traces_experiment_name, when CLEAR runs, then the job fails."""
        payload = build_clear_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            provider_id=clear_provider_id,
            job_name="clear-unknown-traces-experiment",
            parameters={
                "mlflow_traces_experiment_name": CLEAR_UNKNOWN_EXPERIMENT_NAME,
            },
        )
        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        job_id = data["resource"]["id"]
        job_result = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
            timeout=Timeout.TIMEOUT_10MIN,
            sleep=CLEAR_JOB_POLL_INTERVAL,
        )
        state = job_result.get("status", {}).get("state")
        assert state in EVALHUB_JOB_TERMINAL_STATES - {"completed"}, (
            f"Expected failure for unknown traces experiment, got '{state}': {job_result.get('status')}"
        )

    @pytest.mark.tier2
    def test_clear_job_with_experiment_id_parameter(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
        clear_provider_id: str,
        clear_seeded_mlflow_traces_experiment: dict[str, str],
    ) -> None:
        """Given mlflow_traces_experiment_id (no name), when the job is submitted, then API accepts it."""
        payload = build_clear_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            provider_id=clear_provider_id,
            job_name="clear-mlflow-traces-by-id",
            parameters={
                "mlflow_traces_experiment_id": clear_seeded_mlflow_traces_experiment["experiment_id"],
                "mlflow_traces_max_results": 5,
            },
        )
        url = f"https://{evalhub_mt_route.host}{EVALHUB_JOBS_PATH}"
        response = requests.post(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            json=payload,
            verify=evalhub_mt_ca_bundle_file,
            timeout=30,
        )
        assert response.status_code == 202, (
            f"Expected 202 for mlflow_traces_experiment_id job, got {response.status_code}: {response.text}"
        )
