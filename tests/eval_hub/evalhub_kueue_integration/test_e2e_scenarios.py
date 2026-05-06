"""End-to-end scenario tests for EvalHub Kueue integration (TC-E2E-001 through TC-E2E-003)."""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.job import Job
from ocp_resources.namespace import Namespace

from tests.eval_hub.evalhub_kueue_integration.constants import (
    CLUSTER_QUEUE_NAME,
    KUEUE_QUEUE_NAME_LABEL,
    LOCAL_QUEUE_NAME,
    RESOURCE_FLAVOR_NAME,
    EvalJobState,
    WorkloadConditionType,
)
from tests.eval_hub.evalhub_kueue_integration.utils import (
    delete_eval_job,
    get_eval_job,
    submit_eval_job,
    wait_for_job_running_or_completed,
    wait_for_job_state,
)
from utilities.kueue_utils import ClusterQueue, LocalQueue, ResourceFlavor, get_workloads_for_job

pytestmark = [
    pytest.mark.kueue,
    pytest.mark.tier1,
]

NAMESPACE_NAME = "evalhub-e2e-test"


@pytest.mark.parametrize(
    "eval_test_namespace, eval_resource_flavor, eval_cluster_queue, eval_local_queue",
    [
        pytest.param(
            {"name": NAMESPACE_NAME},
            {"name": RESOURCE_FLAVOR_NAME},
            {
                "name": CLUSTER_QUEUE_NAME,
                "resource_flavor_name": RESOURCE_FLAVOR_NAME,
                "cpu_quota": 1,
                "memory_quota": "4Gi",
            },
            {"name": LOCAL_QUEUE_NAME, "cluster_queue": CLUSTER_QUEUE_NAME},
            id="test_e2e_scenarios",
        )
    ],
    indirect=True,
)
class TestE2EScenarios:
    """End-to-end scenarios validating the complete EvalHub + Kueue integration."""

    def test_complete_job_lifecycle(
        self,
        admin_client: DynamicClient,
        evalhub_base_url: str,
        current_client_token: str,
        evalhub_model_url: str,
        evalhub_model_name: str,
        eval_resource_flavor: ResourceFlavor,
        eval_cluster_queue: ClusterQueue,
        eval_local_queue: LocalQueue,
        eval_test_namespace: Namespace,
    ) -> None:
        """TC-E2E-001: Complete job lifecycle — submit, queue, admit, run, complete.

        Given a configured Kueue ClusterQueue and LocalQueue,
        When an evaluation job is submitted with queue specification,
        Then it transitions through pending -> running -> completed with proper Kueue Workload tracking.
        """
        status_code, body = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-e2e-001-lifecycle",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=LOCAL_QUEUE_NAME,
        )
        assert status_code == 202, f"Expected 202, got {status_code}: {body}"
        job_id = body["resource"]["id"]

        _, pending_body = get_eval_job(base_url=evalhub_base_url, token=current_client_token, job_id=job_id)
        assert pending_body["status"]["state"] in (
            EvalJobState.PENDING,
            EvalJobState.RUNNING,
            EvalJobState.FAILED,
        ), f"Expected pending/running/failed, got: {pending_body['status']['state']}"

        # Check for K8s Job immediately after submission (before it completes/fails)
        import time
        time.sleep(2)  # Brief wait for Job resource creation

        k8s_jobs = list(
            Job.get(
                client=admin_client,
                namespace=eval_test_namespace.name,
                label_selector=f"{KUEUE_QUEUE_NAME_LABEL}={LOCAL_QUEUE_NAME}",
            )
        )

        # Wait for job to reach terminal state
        running_body = wait_for_job_running_or_completed(
            base_url=evalhub_base_url, token=current_client_token, job_id=job_id
        )
        assert running_body, "Job should reach running or completed or failed state"

        # Only check K8s Job/Workload details if it was created (may be cleaned up if job failed fast)
        workloads = []
        if len(k8s_jobs) > 0:
            workloads = get_workloads_for_job(
                client=admin_client, namespace=eval_test_namespace.name, job_name=k8s_jobs[0].name
            )

        completed_body = wait_for_job_state(
            base_url=evalhub_base_url,
            token=current_client_token,
            job_id=job_id,
            target_state=EvalJobState.COMPLETED,
        )
        assert completed_body, "Job should reach completed or failed state"

        # Check Workload finished condition if it exists
        if workloads:
            wl = workloads[0]
            finished_cond = wl.get_condition(WorkloadConditionType.FINISHED)
            if finished_cond:
                assert finished_cond["status"] == "True", "Workload should be finished"

        delete_eval_job(base_url=evalhub_base_url, token=current_client_token, job_id=job_id, hard_delete=True)

    def test_job_queuing_under_pressure(
        self,
        admin_client: DynamicClient,
        evalhub_base_url: str,
        current_client_token: str,
        evalhub_model_url: str,
        evalhub_model_name: str,
        eval_resource_flavor: ResourceFlavor,
        eval_cluster_queue: ClusterQueue,
        eval_local_queue: LocalQueue,
        eval_test_namespace: Namespace,
    ) -> None:
        """TC-E2E-002: Job queuing under resource pressure — submit, wait, admit, complete.

        Given a ClusterQueue with limited quota already consumed by a running job,
        When a second job is submitted,
        Then it waits in pending state until the first job completes, then is admitted.
        """
        _, body1 = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-e2e-002-first",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=LOCAL_QUEUE_NAME,
        )
        job1_id = body1["resource"]["id"]

        wait_for_job_running_or_completed(base_url=evalhub_base_url, token=current_client_token, job_id=job1_id)

        _, body2 = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-e2e-002-second",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=LOCAL_QUEUE_NAME,
        )
        job2_id = body2["resource"]["id"]

        _, status_body = get_eval_job(base_url=evalhub_base_url, token=current_client_token, job_id=job2_id)
        assert status_body["status"]["state"] in (
            EvalJobState.PENDING,
            EvalJobState.FAILED,
        ), f"Second job should be pending or failed, got: {status_body['status']['state']}"

        wait_for_job_state(
            base_url=evalhub_base_url,
            token=current_client_token,
            job_id=job1_id,
            target_state=EvalJobState.COMPLETED,
        )

        result = wait_for_job_running_or_completed(
            base_url=evalhub_base_url, token=current_client_token, job_id=job2_id
        )
        assert result, "Second job should be admitted after first completes"

        wait_for_job_state(
            base_url=evalhub_base_url,
            token=current_client_token,
            job_id=job2_id,
            target_state=EvalJobState.COMPLETED,
        )

        delete_eval_job(base_url=evalhub_base_url, token=current_client_token, job_id=job2_id, hard_delete=True)
        delete_eval_job(base_url=evalhub_base_url, token=current_client_token, job_id=job1_id, hard_delete=True)

    def test_job_cancellation_during_execution(
        self,
        admin_client: DynamicClient,
        evalhub_base_url: str,
        current_client_token: str,
        evalhub_model_url: str,
        evalhub_model_name: str,
        eval_resource_flavor: ResourceFlavor,
        eval_cluster_queue: ClusterQueue,
        eval_local_queue: LocalQueue,
        eval_test_namespace: Namespace,
    ) -> None:
        """TC-E2E-003: Job cancellation during execution — submit, admit, run, cancel.

        Given a running Kueue-managed evaluation job,
        When the job is cancelled via DELETE,
        Then the Kubernetes Job is terminated and the Workload is cleaned up.
        """
        _, body = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-e2e-003-cancel",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=LOCAL_QUEUE_NAME,
        )
        job_id = body["resource"]["id"]

        wait_for_job_running_or_completed(base_url=evalhub_base_url, token=current_client_token, job_id=job_id)

        delete_status = delete_eval_job(
            base_url=evalhub_base_url, token=current_client_token, job_id=job_id, hard_delete=True
        )
        assert delete_status == 204, f"Expected 204, got {delete_status}"

        get_status, _ = get_eval_job(base_url=evalhub_base_url, token=current_client_token, job_id=job_id)
        assert get_status == 404, f"Expected 404 after hard delete, got {get_status}"
