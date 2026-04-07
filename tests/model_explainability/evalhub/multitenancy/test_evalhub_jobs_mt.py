import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_resources.service import Service

from tests.model_explainability.evalhub.constants import EVALHUB_JOBS_PATH
from tests.model_explainability.evalhub.utils import (
    build_evalhub_job_payload,
    list_evalhub_jobs,
    submit_evalhub_job,
    validate_evalhub_job_completed,
    validate_evalhub_post_denied,
    validate_evalhub_post_no_tenant,
    validate_evalhub_request_denied,
    validate_evalhub_request_no_tenant,
    wait_for_evalhub_job,
)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-jobs-mt"},
        ),
    ],
    indirect=True,
)
@pytest.mark.model_explainability
class TestEvalHubJobsMT:
    """Multi-tenancy tests for EvalHub job submission.

    Submits an lm_evaluation_harness job (arc_easy, 10 examples)
    against the vLLM emulator. Tests authorization, job lifecycle,
    and result validation.
    """

    def test_job_submit_authorized_tenant(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """User with evaluations-create RBAC in tenant-a can submit a job."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
        )
        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        resource = data.get("resource", {})
        assert resource.get("id"), f"Job response missing 'resource.id': {data}"
        assert data.get("status", {}).get("state") == "pending", (
            f"Expected job state 'pending', got: {data.get('status')}"
        )

    def test_job_completion_and_results(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """Submit a job and wait for it to complete with benchmark results."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
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
            timeout=600,
        )
        validate_evalhub_job_completed(job_data=job_result)

    def test_job_pod_reaches_succeeded(
        self,
        admin_client: DynamicClient,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """After job completion, the K8s pod should be in Succeeded state."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
        )
        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        job_id = data["resource"]["id"]

        # Wait for EvalHub API to report completion
        wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
            timeout=600,
        )

        # Find the job pod by label
        pods = list(
            Pod.get(
                client=admin_client,
                namespace=tenant_a_namespace.name,
                label_selector=f"app=evalhub,component=evaluation-job,job_id={job_id}",
            )
        )
        assert len(pods) >= 1, f"Expected at least 1 pod for job {job_id}, found {len(pods)}"

        # The EvalHub API reports "completed" via sidecar callback before the
        # pod containers actually exit. Wait for the pod to reach Succeeded.
        pod = pods[0]
        pod.wait_for_status(status=Pod.Status.SUCCEEDED, timeout=120)

    def test_job_submit_cross_tenant_denied(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        tenant_b_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """User with RBAC in tenant-a is denied job submission for tenant-b."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
        )
        validate_evalhub_post_denied(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            path=EVALHUB_JOBS_PATH,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_b_namespace.name,
            payload=payload,
        )

    def test_job_submit_missing_tenant_rejected(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """Job submission without X-Tenant header is rejected with 400."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
        )
        validate_evalhub_post_no_tenant(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            path=EVALHUB_JOBS_PATH,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            payload=payload,
        )

    def test_job_list_authorized_tenant(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """After submitting a job, listing jobs for tenant-a shows it."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
        )
        submitted = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        job_id = submitted["resource"]["id"]

        data = list_evalhub_jobs(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
        )
        items = data.get("items", [])
        job_ids = [item.get("resource", {}).get("id") for item in items]
        assert job_id in job_ids, f"Submitted job {job_id} not found in job list: {job_ids}"

    def test_job_list_cross_tenant_denied(
        self,
        tenant_a_token: str,
        tenant_b_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """Listing jobs for tenant-b is denied for user with tenant-a access."""
        validate_evalhub_request_denied(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            path=EVALHUB_JOBS_PATH,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_b_namespace.name,
        )

    def test_job_list_missing_tenant_rejected(
        self,
        tenant_a_token: str,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """Listing jobs without X-Tenant header is rejected with 400."""
        validate_evalhub_request_no_tenant(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            path=EVALHUB_JOBS_PATH,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
        )
