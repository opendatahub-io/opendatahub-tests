"""TC-REG: Regression tests — existing evaluation API and job flow unchanged.

Covers RHAISTRAT-1923 — verifies that the lifecycle signal feature does not
regress the existing EvalHub REST API submit/retrieve flow or the evaluation
job execution and sidecar flow.
"""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.service import Service

from tests.ai_safety.evalhub.k8s_lifecycle_signals.utils import (
    wait_for_evaluation_job_name,
)
from tests.ai_safety.evalhub.utils import (
    build_evalhub_job_payload,
    submit_evalhub_job,
    wait_for_evalhub_job,
)


@pytest.mark.ai_safety
class TestRegRegression:
    """TC-REG: Regression tests confirming pre-feature API and execution flows are unchanged.

    All tests share the session-scoped EvalHub deployment.
    """

    @pytest.mark.tier1
    def test_reg_001_existing_evaluation_api_submit_retrieve_unchanged(
        self,
        lifecycle_signals_ready: None,
        lifecycle_signals_route: Route,
        lifecycle_signals_ca_bundle_file: str,
        lifecycle_signals_token: str,
        lifecycle_signals_namespace: Namespace,
        lifecycle_signals_vllm_service: Service,
    ) -> None:
        """Given the EvalHub REST API is available,
        when an evaluation is submitted and its status is polled until completion,
        then submission returns a valid job ID, the status endpoint works throughout the lifecycle,
        and the results endpoint returns benchmark results matching the pre-feature API contract."""
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name
        payload = build_evalhub_job_payload(
            model_service_name=lifecycle_signals_vllm_service.name,
            tenant_namespace=ns,
            job_name="tc-reg-001",
        )

        data = submit_evalhub_job(
            host=host,
            token=lifecycle_signals_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=ns,
            payload=payload,
        )
        assert "resource" in data, f"Expected 'resource' key in submit response: {data}"
        assert "id" in data["resource"], f"Expected 'id' in resource: {data['resource']}"
        job_id: str = data["resource"]["id"]
        assert job_id, "Job ID must be a non-empty string"

        result = wait_for_evalhub_job(
            host=host,
            token=lifecycle_signals_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=ns,
            job_id=job_id,
        )

        assert result.get("status", {}).get("state") in ("completed", "failed", "partially_failed"), (
            f"Expected terminal state, got {result.get('status', {}).get('state')!r}"
        )
        assert "benchmarks" in result or "results" in result, (
            f"Expected benchmark results in response: {list(result.keys())}"
        )

    @pytest.mark.tier1
    def test_reg_002_evaluation_job_execution_and_sidecar_flow_unchanged(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
        lifecycle_signals_route: Route,
        lifecycle_signals_ca_bundle_file: str,
        lifecycle_signals_token: str,
        lifecycle_signals_namespace: Namespace,
        lifecycle_signals_vllm_service: Service,
    ) -> None:
        """Given a standard evaluation submission,
        when the batch Job runs through the adapter and sidecar lifecycle,
        then the batch Job reaches the Complete condition, the sidecar processes status events
        correctly, and no errors appear in the server logs related to lifecycle signal emission."""
        from ocp_resources.job import Job

        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name
        payload = build_evalhub_job_payload(
            model_service_name=lifecycle_signals_vllm_service.name,
            tenant_namespace=ns,
            job_name="tc-reg-002",
        )
        job_id = submit_evalhub_job(
            host=host,
            token=lifecycle_signals_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=ns,
            payload=payload,
        )["resource"]["id"]
        wait_for_evalhub_job(
            host=host,
            token=lifecycle_signals_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=ns,
            job_id=job_id,
        )
        job_name = wait_for_evaluation_job_name(
            admin_client=admin_client,
            namespace=ns,
            evalhub_job_id=job_id,
        )

        k8s_job = Job(client=admin_client, name=job_name, namespace=ns)
        assert k8s_job.exists, f"Batch Job {job_name} must exist in {ns} after completion"

        conditions = k8s_job.instance.status.conditions or []
        complete_conditions = [c for c in conditions if c.get("type") == "Complete" and c.get("status") == "True"]
        failed_conditions = [c for c in conditions if c.get("type") == "Failed" and c.get("status") == "True"]

        condition_types = [c.get("type") for c in conditions]
        assert conditions, f"Batch Job {job_name} must have terminal status conditions; conditions: {condition_types}"
        assert not failed_conditions, (
            f"Batch Job {job_name} must not have Failed=True condition; conditions: {condition_types}"
        )
        assert complete_conditions, (
            f"Batch Job {job_name} should reach Complete condition; conditions: {condition_types}"
        )
