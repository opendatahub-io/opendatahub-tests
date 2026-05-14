import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.job import Job
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.service import Service
from timeout_sampler import TimeoutSampler

from tests.model_explainability.evalhub.utils import (
    _get_job_status,
    build_evalhub_job_payload,
    check_workload_quota_reserved,
    evalhub_runtime_label_selector,
    submit_evalhub_job,
    validate_evalhub_job_completed,
    wait_for_evalhub_job,
    wait_for_evalhub_job_workload_admitted,
    wait_for_evalhub_job_workload_inadmissible,
)
from utilities.kueue_utils import LocalQueue, check_gated_pods_and_running_pods

@pytest.mark.tier1
@pytest.mark.kueue
@pytest.mark.model_explainability
class TestEvalHubKueueBasic:
    """Basic lifecycle tests for EvalHub jobs with Kueue admission control."""

    def test_evalhub_job_workload_created(
        self,
        admin_client: DynamicClient,
        evalhub_kueue_namespace: Namespace,
        evalhub_kueue_multi_job_local_queue: LocalQueue,
        evalhub_kueue_user_token: str,
        evalhub_kueue_vllm_service: Service,
        evalhub_mt_route: Route,
        evalhub_mt_ca_bundle_file: str,
    ) -> None:
        """Submit an EvalHub job and verify Kueue Workload is created."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_kueue_vllm_service.name,
            tenant_namespace=evalhub_kueue_namespace.name,
        )

        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=evalhub_kueue_user_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=evalhub_kueue_namespace.name,
            payload=payload,
        )
        job_id = data["resource"]["id"]

        # Wait for Kueue Workload to be created and admitted
        workload = wait_for_evalhub_job_workload_admitted(
            admin_client=admin_client,
            namespace=evalhub_kueue_namespace.name,
            evalhub_job_id=job_id,
            timeout=120,
        )

        assert workload is not None
        assert check_workload_quota_reserved(workload)

    def test_evalhub_job_lifecycle_with_kueue(
        self,
        admin_client: DynamicClient,
        evalhub_kueue_namespace: Namespace,
        evalhub_kueue_multi_job_local_queue: LocalQueue,
        evalhub_kueue_user_token: str,
        evalhub_kueue_vllm_service: Service,
        evalhub_mt_route: Route,
        evalhub_mt_ca_bundle_file: str,
    ) -> None:
        """Full lifecycle: submit → admitted → running → completed."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_kueue_vllm_service.name,
            tenant_namespace=evalhub_kueue_namespace.name,
        )

        # 1. Submit job
        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=evalhub_kueue_user_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=evalhub_kueue_namespace.name,
            payload=payload,
        )
        job_id = data["resource"]["id"]

        # 2. Verify Kueue admits the workload
        wait_for_evalhub_job_workload_admitted(
            admin_client=admin_client,
            namespace=evalhub_kueue_namespace.name,
            evalhub_job_id=job_id,
        )

        # 3. Verify pod is running (not gated)
        selector = evalhub_runtime_label_selector(evalhub_job_id=job_id)
        running_pods, gated_pods = check_gated_pods_and_running_pods(
            labels=[selector],
            namespace=evalhub_kueue_namespace.name,
            admin_client=admin_client,
        )
        assert running_pods >= 1, f"Expected >=1 running pod, got {running_pods}"
        assert gated_pods == 0, f"Expected 0 gated pods, got {gated_pods}"

        # 4. Wait for EvalHub API to report completion
        job_result = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=evalhub_kueue_user_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=evalhub_kueue_namespace.name,
            job_id=job_id,
            timeout=600,
        )
        validate_evalhub_job_completed(job_data=job_result)

        # 5. Verify K8s Job succeeded
        jobs = list(
            Job.get(
                client=admin_client,
                namespace=evalhub_kueue_namespace.name,
                label_selector=selector,
            )
        )
        assert len(jobs) >= 1
        # Job should have completions=1
        job = jobs[0]
        job.wait_for_condition(condition="Complete", status="True", timeout=120)

    def test_evalhub_status_reflects_kueue_state(
        self,
        admin_client: DynamicClient,
        evalhub_kueue_namespace: Namespace,
        evalhub_kueue_multi_job_local_queue: LocalQueue,
        evalhub_kueue_user_token: str,
        evalhub_kueue_vllm_service: Service,
        evalhub_mt_route: Route,
        evalhub_mt_ca_bundle_file: str,
    ) -> None:
        """EvalHub API status should transition: pending → running → completed."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_kueue_vllm_service.name,
            tenant_namespace=evalhub_kueue_namespace.name,
        )

        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=evalhub_kueue_user_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=evalhub_kueue_namespace.name,
            payload=payload,
        )
        job_id = data["resource"]["id"]

        # Initial state should be pending
        assert data["status"]["state"] == "pending"

        # Wait for admission and running state
        for sample in TimeoutSampler(
            wait_timeout=180,
            sleep=10,
            func=_get_job_status,
            host=evalhub_mt_route.host,
            token=evalhub_kueue_user_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=evalhub_kueue_namespace.name,
            job_id=job_id,
        ):
            state = sample.get("status", {}).get("state", "")
            if state == "running":
                break

        # Wait for completion
        job_result = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=evalhub_kueue_user_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=evalhub_kueue_namespace.name,
            job_id=job_id,
            timeout=600,
        )
        assert job_result["status"]["state"] == "completed"

    def test_queue_capacity_exhaustion(
        self,
        admin_client: DynamicClient,
        evalhub_kueue_namespace: Namespace,
        evalhub_kueue_single_job_local_queue: LocalQueue,
        evalhub_kueue_user_token: str,
        evalhub_kueue_vllm_service: Service,
        evalhub_mt_route: Route,
        evalhub_mt_ca_bundle_file: str,
    ) -> None:
        """Validate failure behavior: job rejection when queue capacity is exhausted.

        Submit 2 jobs with quota for 1, verify second is gated.
        """
        payload1 = build_evalhub_job_payload(
            model_service_name=evalhub_kueue_vllm_service.name,
            tenant_namespace=evalhub_kueue_namespace.name,
        )

        # Submit first job
        data1 = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=evalhub_kueue_user_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=evalhub_kueue_namespace.name,
            payload=payload1,
        )
        job1_id = data1["resource"]["id"]

        # Wait for first job to be admitted
        wait_for_evalhub_job_workload_admitted(
            admin_client=admin_client,
            namespace=evalhub_kueue_namespace.name,
            evalhub_job_id=job1_id,
        )

        # Submit second job
        payload2 = build_evalhub_job_payload(
            model_service_name=evalhub_kueue_vllm_service.name,
            tenant_namespace=evalhub_kueue_namespace.name,
        )
        data2 = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=evalhub_kueue_user_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=evalhub_kueue_namespace.name,
            payload=payload2,
        )
        job2_id = data2["resource"]["id"]

        # Verify second job's workload is inadmissible
        wait_for_evalhub_job_workload_inadmissible(
            admin_client=admin_client,
            namespace=evalhub_kueue_namespace.name,
            evalhub_job_id=job2_id,
        )

        # Verify pod counts: 1 running, 1 gated
        all_labels = f"app=evalhub,component=evaluation-job"
        running_pods, gated_pods = check_gated_pods_and_running_pods(
            labels=[all_labels],
            namespace=evalhub_kueue_namespace.name,
            admin_client=admin_client,
        )

        assert running_pods == 1, f"Expected 1 running pod, got {running_pods}"
        assert gated_pods == 1, f"Expected 1 gated pod, got {gated_pods}"
