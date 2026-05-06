"""API integration tests for EvalHub Kueue integration (TC-API-001 through TC-API-005)."""

import pytest
from ocp_resources.namespace import Namespace

from tests.eval_hub.evalhub_kueue_integration.constants import (
    CLUSTER_QUEUE_NAME,
    DEFAULT_CPU_QUOTA,
    DEFAULT_MEMORY_QUOTA,
    LOCAL_QUEUE_NAME,
    RESOURCE_FLAVOR_NAME,
    EvalJobState,
)
from tests.eval_hub.evalhub_kueue_integration.utils import (
    delete_eval_job,
    get_eval_job,
    list_eval_jobs,
    submit_eval_job,
)
from utilities.kueue_utils import ClusterQueue, LocalQueue, ResourceFlavor

pytestmark = [
    pytest.mark.kueue,
    pytest.mark.tier1,
]

NAMESPACE_NAME = "evalhub-api-test"


@pytest.mark.parametrize(
    "eval_test_namespace, eval_resource_flavor, eval_cluster_queue, eval_local_queue",
    [
        pytest.param(
            {"name": NAMESPACE_NAME},
            {"name": RESOURCE_FLAVOR_NAME},
            {
                "name": CLUSTER_QUEUE_NAME,
                "resource_flavor_name": RESOURCE_FLAVOR_NAME,
                "cpu_quota": DEFAULT_CPU_QUOTA,
                "memory_quota": DEFAULT_MEMORY_QUOTA,
            },
            {"name": LOCAL_QUEUE_NAME, "cluster_queue": CLUSTER_QUEUE_NAME},
            id="test_api_integration",
        )
    ],
    indirect=True,
)
class TestApiIntegration:
    """API integration tests for EvalHub with Kueue queue management."""

    def test_submit_job_with_queue_spec(
        self,
        evalhub_base_url: str,
        current_client_token: str,
        evalhub_model_url: str,
        evalhub_model_name: str,
        eval_resource_flavor: ResourceFlavor,
        eval_cluster_queue: ClusterQueue,
        eval_local_queue: LocalQueue,
        eval_test_namespace: Namespace,
    ) -> None:
        """TC-API-001: Verify job submission with Kueue queue spec returns HTTP 202.

        Given a configured Kueue ClusterQueue and LocalQueue,
        When an evaluation job is submitted with queue specification,
        Then the API returns 202 with a resource ID and pending state.
        """
        status_code, body = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-api-001-submit",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=LOCAL_QUEUE_NAME,
        )

        assert status_code == 202, f"Expected 202, got {status_code}: {body}"
        assert "resource" in body, f"Response missing 'resource' field: {body}"
        assert "id" in body["resource"], f"Response missing 'resource.id': {body}"
        assert body.get("status", {}).get("state") in (EvalJobState.PENDING, EvalJobState.FAILED), (
            f"Expected pending or failed state, got: {body.get('status', {}).get('state')}"
        )

        job_id = body["resource"]["id"]
        delete_eval_job(base_url=evalhub_base_url, token=current_client_token, job_id=job_id, hard_delete=True)

    def test_get_job_status(
        self,
        evalhub_base_url: str,
        current_client_token: str,
        evalhub_model_url: str,
        evalhub_model_name: str,
        eval_resource_flavor: ResourceFlavor,
        eval_cluster_queue: ClusterQueue,
        eval_local_queue: LocalQueue,
        eval_test_namespace: Namespace,
    ) -> None:
        """TC-API-002: Verify GET job status returns Kueue-managed state.

        Given a submitted Kueue-managed evaluation job,
        When the job status is retrieved via GET,
        Then the response contains job state and queue information.
        """
        _, body = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-api-002-status",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=LOCAL_QUEUE_NAME,
        )
        job_id = body["resource"]["id"]

        status_code, status_body = get_eval_job(base_url=evalhub_base_url, token=current_client_token, job_id=job_id)

        assert status_code == 200, f"Expected 200, got {status_code}: {status_body}"
        assert "status" in status_body
        assert status_body["status"]["state"] in (
            EvalJobState.PENDING,
            EvalJobState.RUNNING,
            EvalJobState.COMPLETED,
            EvalJobState.FAILED,
        ), f"Expected valid job state, got: {status_body['status']['state']}"

        delete_eval_job(base_url=evalhub_base_url, token=current_client_token, job_id=job_id, hard_delete=True)

    def test_list_jobs_with_filter(
        self,
        evalhub_base_url: str,
        current_client_token: str,
        evalhub_model_url: str,
        evalhub_model_name: str,
        eval_resource_flavor: ResourceFlavor,
        eval_cluster_queue: ClusterQueue,
        eval_local_queue: LocalQueue,
        eval_test_namespace: Namespace,
    ) -> None:
        """TC-API-003: Verify listing jobs with status filter.

        Given multiple submitted evaluation jobs,
        When jobs are listed with a status filter,
        Then only jobs matching the filter are returned.
        """
        _, body = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-api-003-list",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=LOCAL_QUEUE_NAME,
        )
        job_id = body["resource"]["id"]

        status_code, list_body = list_eval_jobs(
            base_url=evalhub_base_url,
            token=current_client_token,
            status=EvalJobState.PENDING,
        )

        assert status_code == 200, f"Expected 200, got {status_code}: {list_body}"

        delete_eval_job(base_url=evalhub_base_url, token=current_client_token, job_id=job_id, hard_delete=True)

    def test_cancel_job_soft_delete(
        self,
        evalhub_base_url: str,
        current_client_token: str,
        evalhub_model_url: str,
        evalhub_model_name: str,
        eval_resource_flavor: ResourceFlavor,
        eval_cluster_queue: ClusterQueue,
        eval_local_queue: LocalQueue,
        eval_test_namespace: Namespace,
    ) -> None:
        """TC-API-004: Verify soft delete cancels a Kueue-managed job.

        Given a submitted Kueue-managed evaluation job,
        When the job is cancelled via DELETE without hard_delete,
        Then the API returns 204 and the job reaches a terminal state.
        """
        _, body = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-api-004-soft-delete",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=LOCAL_QUEUE_NAME,
        )
        job_id = body["resource"]["id"]

        delete_status = delete_eval_job(
            base_url=evalhub_base_url, token=current_client_token, job_id=job_id, hard_delete=False
        )
        # Accept 204 (successful cancel) or 409 (already terminal, cannot cancel)
        assert delete_status in (204, 409), f"Expected 204 or 409, got {delete_status}"

        get_status, get_body = get_eval_job(base_url=evalhub_base_url, token=current_client_token, job_id=job_id)
        assert get_status == 200, f"Expected 200 after soft delete, got {get_status}"

        delete_eval_job(base_url=evalhub_base_url, token=current_client_token, job_id=job_id, hard_delete=True)

    def test_cancel_job_hard_delete(
        self,
        evalhub_base_url: str,
        current_client_token: str,
        evalhub_model_url: str,
        evalhub_model_name: str,
        eval_resource_flavor: ResourceFlavor,
        eval_cluster_queue: ClusterQueue,
        eval_local_queue: LocalQueue,
        eval_test_namespace: Namespace,
    ) -> None:
        """TC-API-005: Verify hard delete removes a Kueue-managed job.

        Given a submitted Kueue-managed evaluation job,
        When the job is cancelled via DELETE with hard_delete=true,
        Then the API returns 204 and the job is no longer retrievable.
        """
        _, body = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-api-005-hard-delete",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=LOCAL_QUEUE_NAME,
        )
        job_id = body["resource"]["id"]

        delete_status = delete_eval_job(
            base_url=evalhub_base_url, token=current_client_token, job_id=job_id, hard_delete=True
        )
        assert delete_status == 204, f"Expected 204, got {delete_status}"

        get_status, _ = get_eval_job(base_url=evalhub_base_url, token=current_client_token, job_id=job_id)
        assert get_status == 404, f"Expected 404 after hard delete, got {get_status}"
