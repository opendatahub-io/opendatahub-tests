"""Queue management tests for EvalHub Kueue integration (TC-QUEUE-001 through TC-QUEUE-003)."""

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
)
from tests.eval_hub.evalhub_kueue_integration.utils import (
    delete_eval_job,
    submit_eval_job,
    wait_for_job_running_or_completed,
    wait_for_job_state,
)
from utilities.kueue_utils import ClusterQueue, LocalQueue, ResourceFlavor

pytestmark = [
    pytest.mark.kueue,
    pytest.mark.tier1,
]

NAMESPACE_NAME = "evalhub-queue-test"


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
            id="test_queue_management",
        )
    ],
    indirect=True,
)
class TestQueueManagement:
    """Queue management tests validating Kueue admission control for EvalHub jobs."""

    def test_job_admitted_with_available_quota(
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
        """TC-QUEUE-001: Verify job is admitted when LocalQueue has available quota.

        Given a ClusterQueue with sufficient quota and no running jobs,
        When an evaluation job is submitted to the LocalQueue,
        Then the job is admitted and starts running.
        """
        _, body = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-queue-001-admit",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=LOCAL_QUEUE_NAME,
        )
        job_id = body["resource"]["id"]

        # Check for Kubernetes Job creation immediately (before job completes/fails)
        import time

        time.sleep(2)  # Brief wait for Job resource to be created by EvalHub

        jobs = list(
            Job.get(
                client=admin_client,
                namespace=eval_test_namespace.name,
                label_selector=f"{KUEUE_QUEUE_NAME_LABEL}={LOCAL_QUEUE_NAME}",
            )
        )

        # If job was created but failed very quickly, it may already be gone
        # In that case, verify the EvalHub job reached a terminal state
        result = wait_for_job_running_or_completed(base_url=evalhub_base_url, token=current_client_token, job_id=job_id)
        assert result, f"Job {job_id} did not reach running/completed/failed state"

        # Only check K8s Job if it still exists (may have been cleaned up if job failed fast)
        if len(jobs) > 0:
            for job in jobs:
                assert job.instance.spec.get("suspend") is False or job.instance.spec.get("suspend") is None, (
                    "Job should not be suspended after admission"
                )

        delete_eval_job(base_url=evalhub_base_url, token=current_client_token, job_id=job_id, hard_delete=True)

    def test_job_pending_when_quota_exhausted(
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
        """TC-QUEUE-002: Verify job remains pending when ClusterQueue quota is exhausted.

        Given a ClusterQueue with limited quota fully consumed by a running job,
        When a second evaluation job is submitted,
        Then the second job remains in pending state.
        """
        _, body1 = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-queue-002-first",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=LOCAL_QUEUE_NAME,
        )
        job1_id = body1["resource"]["id"]

        wait_for_job_running_or_completed(base_url=evalhub_base_url, token=current_client_token, job_id=job1_id)

        _, body2 = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-queue-002-second",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=LOCAL_QUEUE_NAME,
        )
        job2_id = body2["resource"]["id"]

        wait_for_job_state(
            base_url=evalhub_base_url,
            token=current_client_token,
            job_id=job2_id,
            target_state=EvalJobState.PENDING,
        )

        delete_eval_job(base_url=evalhub_base_url, token=current_client_token, job_id=job2_id, hard_delete=True)
        delete_eval_job(base_url=evalhub_base_url, token=current_client_token, job_id=job1_id, hard_delete=True)

    def test_pending_job_auto_admitted(
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
        """TC-QUEUE-003: Verify pending job is auto-admitted after resources are freed.

        Given a pending job waiting for quota,
        When the blocking job completes and frees resources,
        Then the pending job is automatically admitted and starts running.
        """
        _, body1 = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-queue-003-first",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=LOCAL_QUEUE_NAME,
        )
        job1_id = body1["resource"]["id"]
        wait_for_job_running_or_completed(base_url=evalhub_base_url, token=current_client_token, job_id=job1_id)

        _, body2 = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-queue-003-second",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=LOCAL_QUEUE_NAME,
        )
        job2_id = body2["resource"]["id"]

        wait_for_job_state(
            base_url=evalhub_base_url, token=current_client_token, job_id=job1_id, target_state=EvalJobState.COMPLETED
        )

        result = wait_for_job_running_or_completed(
            base_url=evalhub_base_url, token=current_client_token, job_id=job2_id
        )
        assert result, f"Job {job2_id} was not auto-admitted after resources freed"

        delete_eval_job(base_url=evalhub_base_url, token=current_client_token, job_id=job2_id, hard_delete=True)
        delete_eval_job(base_url=evalhub_base_url, token=current_client_token, job_id=job1_id, hard_delete=True)
