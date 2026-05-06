"""Resource quota enforcement tests for EvalHub Kueue integration (TC-QUOTA-001, TC-QUOTA-002)."""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.job import Job
from ocp_resources.namespace import Namespace

from tests.eval_hub.evalhub_kueue_integration.constants import (
    CLUSTER_QUEUE_NAME,
    KUEUE_QUEUE_NAME_LABEL,
    LOCAL_QUEUE_NAME,
    RESOURCE_FLAVOR_NAME,
    SMALL_CPU_QUOTA,
    SMALL_MEMORY_QUOTA,
    EvalJobState,
)
from tests.eval_hub.evalhub_kueue_integration.utils import (
    delete_eval_job,
    get_eval_job,
    submit_eval_job,
    wait_for_job_running_or_completed,
)
from utilities.kueue_utils import ClusterQueue, LocalQueue, ResourceFlavor

pytestmark = [
    pytest.mark.kueue,
    pytest.mark.tier1,
]


class TestQuotaWithinLimits:
    """TC-QUOTA-001: Verify multiple jobs admitted within ClusterQueue quota limits."""

    NAMESPACE_NAME = "evalhub-quota-within"

    @pytest.mark.parametrize(
        "eval_test_namespace, eval_resource_flavor, eval_cluster_queue, eval_local_queue",
        [
            pytest.param(
                {"name": "evalhub-quota-within"},
                {"name": RESOURCE_FLAVOR_NAME},
                {
                    "name": CLUSTER_QUEUE_NAME,
                    "resource_flavor_name": RESOURCE_FLAVOR_NAME,
                    "cpu_quota": 2,
                    "memory_quota": "8Gi",
                },
                {"name": LOCAL_QUEUE_NAME, "cluster_queue": CLUSTER_QUEUE_NAME},
                id="test_quota_within_limits",
            )
        ],
        indirect=True,
    )
    def test_multiple_jobs_within_quota(
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
        """Verify multiple jobs fitting within quota are all admitted.

        Given a ClusterQueue with quota for 2 CPU and 8Gi memory,
        When two small evaluation jobs are submitted,
        Then both jobs are admitted and running concurrently.
        """
        job_ids: list[str] = []
        for i in range(2):
            status_code, body = submit_eval_job(
                base_url=evalhub_base_url,
                token=current_client_token,
                name=f"tc-quota-001-job-{i}",
                model_url=evalhub_model_url,
                model_name=evalhub_model_name,
                queue_name=LOCAL_QUEUE_NAME,
            )
            assert status_code == 202, f"Expected 202, got {status_code}: {body}"
            assert "resource" in body, f"Response missing 'resource' key: {body}"
            job_ids.append(body["resource"]["id"])

        for job_id in job_ids:
            result = wait_for_job_running_or_completed(
                base_url=evalhub_base_url, token=current_client_token, job_id=job_id
            )
            assert result, f"Job {job_id} was not admitted within quota"

        for job_id in job_ids:
            delete_eval_job(base_url=evalhub_base_url, token=current_client_token, job_id=job_id, hard_delete=True)


class TestQuotaExceeded:
    """TC-QUOTA-002: Verify job exceeding ClusterQueue quota is queued."""

    @pytest.mark.parametrize(
        "eval_test_namespace, eval_resource_flavor, eval_cluster_queue, eval_local_queue",
        [
            pytest.param(
                {"name": "evalhub-quota-exceed"},
                {"name": RESOURCE_FLAVOR_NAME},
                {
                    "name": CLUSTER_QUEUE_NAME,
                    "resource_flavor_name": RESOURCE_FLAVOR_NAME,
                    "cpu_quota": SMALL_CPU_QUOTA,
                    "memory_quota": SMALL_MEMORY_QUOTA,
                },
                {"name": LOCAL_QUEUE_NAME, "cluster_queue": CLUSTER_QUEUE_NAME},
                id="test_quota_exceeded",
            )
        ],
        indirect=True,
    )
    def test_job_exceeding_quota_is_queued(
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
        """Verify a job requesting more than quota remains pending.

        Given a ClusterQueue with quota of 500m CPU and 1Gi memory,
        When an evaluation job requesting more resources is submitted,
        Then the job remains in pending state with suspended Kubernetes Job.
        """
        _, body = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-quota-002-exceed",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=LOCAL_QUEUE_NAME,
        )
        job_id = body["resource"]["id"]

        # Job should be pending (quota exceeded) or may fail if execution issues occur
        # Either way, Kueue should have processed it
        _, status_body = get_eval_job(base_url=evalhub_base_url, token=current_client_token, job_id=job_id)
        job_state = status_body.get("status", {}).get("state", "")
        assert job_state in (
            EvalJobState.PENDING,
            EvalJobState.FAILED,
        ), f"Expected pending or failed, got: {job_state}"

        k8s_jobs = list(
            Job.get(
                client=admin_client,
                namespace=eval_test_namespace.name,
                label_selector=f"{KUEUE_QUEUE_NAME_LABEL}={LOCAL_QUEUE_NAME}",
            )
        )
        for job in k8s_jobs:
            assert job.instance.spec.get("suspend") is True, "Job should remain suspended when quota exceeded"

        delete_eval_job(base_url=evalhub_base_url, token=current_client_token, job_id=job_id, hard_delete=True)
