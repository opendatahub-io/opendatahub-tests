"""CLEAR evaluation with PVC-staged agent traces (RHOAIENG-63558 baseline path).

Given: PVC containing CLEAR-compatible *.json agent traces
When:  A CLEAR job is submitted with test_data_ref.pvc (no mlflow_traces_*)
Then:  The job completes and returns CLEAR metrics
"""

from __future__ import annotations

import pytest
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.route import Route
from ocp_resources.service import Service

from tests.ai_safety.evalhub.clear.constants import (
    CLEAR_JOB_POLL_INTERVAL,
    CLEAR_JOB_TIMEOUT,
    CLEAR_RESULTS_EXPERIMENT_NAME,
    CLEAR_TRACES_SUB_PATH,
)
from tests.ai_safety.evalhub.clear.utils import build_clear_job_payload, validate_clear_job_completed
from tests.ai_safety.evalhub.utils import (
    build_pvc_test_data_ref,
    submit_evalhub_job,
    wait_for_evalhub_job,
)


@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-evalhub-clear-pvc"})],
    indirect=True,
)
@pytest.mark.ai_safety
@pytest.mark.usefixtures("evalhub_mt_ready")
class TestEvalHubClearPvcTraces:
    """C-09: CLEAR agentic evaluation using PVC-staged JSON traces."""

    @pytest.mark.tier1
    @pytest.mark.slow
    def test_clear_job_with_pvc_traces_completes(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
        clear_provider_id: str,
        clear_traces_pvc_populated: PersistentVolumeClaim,
    ) -> None:
        """Given PVC-staged agent traces, when CLEAR agentic-evaluation runs, then it completes."""
        payload = build_clear_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            provider_id=clear_provider_id,
            job_name="clear-pvc-traces-e2e",
            experiment_name=CLEAR_RESULTS_EXPERIMENT_NAME,
            test_data_ref=build_pvc_test_data_ref(
                claim_name=clear_traces_pvc_populated.name,
                sub_path=CLEAR_TRACES_SUB_PATH,
            ),
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
            timeout=CLEAR_JOB_TIMEOUT,
            sleep=CLEAR_JOB_POLL_INTERVAL,
        )
        validate_clear_job_completed(job_data=job_result, provider_id=clear_provider_id)

    @pytest.mark.tier2
    def test_clear_job_pvc_traces_with_parameter_mlflow_experiment_fallback(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
        clear_provider_id: str,
        clear_traces_pvc_populated: PersistentVolumeClaim,
    ) -> None:
        """Given no job experiment.name but parameters.mlflow_experiment_name, submit is accepted.

        CLEAR copies parameters.mlflow_experiment_name onto JobSpec before callbacks.mlflow.save()
        when experiment_name is unset (results-upload fallback).
        """
        payload = build_clear_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            provider_id=clear_provider_id,
            job_name="clear-pvc-param-mlflow-fallback",
            test_data_ref=build_pvc_test_data_ref(
                claim_name=clear_traces_pvc_populated.name,
                sub_path=CLEAR_TRACES_SUB_PATH,
            ),
            parameters={
                "mlflow_experiment_name": "odh-fvt-clear-param-fallback-results",
            },
        )
        assert "experiment" not in payload
        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        assert data.get("resource", {}).get("id"), f"Expected job id, got: {data}"
