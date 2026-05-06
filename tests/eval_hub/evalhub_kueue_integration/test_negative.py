"""Negative tests for EvalHub Kueue integration (TC-NEG-001 through TC-NEG-005)."""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.job import Job
from ocp_resources.namespace import Namespace

from tests.eval_hub.evalhub_kueue_integration.constants import (
    CLUSTER_QUEUE_NAME,
    DEFAULT_CPU_QUOTA,
    DEFAULT_MEMORY_QUOTA,
    KUEUE_QUEUE_NAME_LABEL,
    LOCAL_QUEUE_NAME,
    RESOURCE_FLAVOR_NAME,
    EvalJobState,
)
from tests.eval_hub.evalhub_kueue_integration.utils import (
    delete_eval_job,
    get_eval_job,
    submit_eval_job,
)
from utilities.kueue_utils import ClusterQueue, LocalQueue, ResourceFlavor, Workload

pytestmark = [
    pytest.mark.kueue,
]

NAMESPACE_NAME = "evalhub-neg-test"


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
            id="test_negative_scenarios",
        )
    ],
    indirect=True,
)
class TestNegativeScenarios:
    """Negative tests for EvalHub Kueue integration."""

    @pytest.mark.tier1
    def test_nonexistent_queue_name(
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
        """TC-NEG-001: Verify error when submitting job with non-existent queue name.

        Given a Kueue-enabled cluster with no LocalQueue named 'nonexistent-queue',
        When a job is submitted referencing that queue,
        Then an error response is returned or the job shows admission failure.
        """
        status_code, body = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-neg-001-invalid-queue",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name="nonexistent-queue",
        )

        if status_code == 400:
            assert "error" in str(body).lower() or "message" in body
        elif status_code == 202:
            job_id = body["resource"]["id"]
            _, status_body = get_eval_job(base_url=evalhub_base_url, token=current_client_token, job_id=job_id)
            state = status_body.get("status", {}).get("state")
            assert state in (EvalJobState.PENDING, EvalJobState.FAILED), (
                f"Job with invalid queue should be pending or failed, got {state}"
            )
            delete_eval_job(base_url=evalhub_base_url, token=current_client_token, job_id=job_id, hard_delete=True)
        else:
            pytest.fail(f"Unexpected status code {status_code} for non-existent queue: {body}")

    @pytest.mark.tier2
    @pytest.mark.pre_upgrade
    @pytest.mark.post_upgrade
    def test_submit_without_queue_spec(
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
        """TC-NEG-002: Verify job without queue spec runs without Kueue (backwards compatibility).

        Given EvalHub deployed with or without Kueue,
        When a job is submitted without the queue field,
        Then the job is accepted (202) and runs without Kueue management.
        """
        status_code, body = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-neg-002-no-queue",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=None,
        )

        assert status_code == 202, f"Expected 202, got {status_code}: {body}"
        job_id = body["resource"]["id"]

        jobs_with_kueue_label = list(
            Job.get(
                client=admin_client,
                namespace=eval_test_namespace.name,
                label_selector=KUEUE_QUEUE_NAME_LABEL,
            )
        )
        job_names_with_label = {j.name for j in jobs_with_kueue_label}

        workloads = list(Workload.get(client=admin_client, namespace=eval_test_namespace.name))
        assert not any(wl.instance.get("metadata", {}).get("name", "").startswith("tc-neg-002") for wl in workloads), (
            "No Workload should be created for job without queue spec"
        )

        delete_eval_job(base_url=evalhub_base_url, token=current_client_token, job_id=job_id, hard_delete=True)

    @pytest.mark.tier1
    def test_unauthorized_returns_401(
        self,
        evalhub_base_url: str,
        evalhub_model_url: str,
        evalhub_model_name: str,
        eval_resource_flavor: ResourceFlavor,
        eval_cluster_queue: ClusterQueue,
        eval_local_queue: LocalQueue,
        eval_test_namespace: Namespace,
    ) -> None:
        """TC-NEG-003: Verify unauthorized request returns 401.

        Given EvalHub requires authentication,
        When a request is made without a valid bearer token,
        Then the API returns 401 Unauthorized.
        """
        status_code, body = submit_eval_job(
            base_url=evalhub_base_url,
            token="invalid-token-12345",
            name="tc-neg-003-unauth",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=LOCAL_QUEUE_NAME,
        )

        assert status_code == 401, f"Expected 401, got {status_code}: {body}"

    @pytest.mark.tier2
    def test_forbidden_returns_403(
        self,
        evalhub_base_url: str,
        evalhub_model_url: str,
        evalhub_model_name: str,
        eval_resource_flavor: ResourceFlavor,
        eval_cluster_queue: ClusterQueue,
        eval_local_queue: LocalQueue,
        eval_test_namespace: Namespace,
    ) -> None:
        """TC-NEG-004: Verify forbidden request returns 403.

        Given a valid but insufficiently privileged user,
        When the user attempts to submit an evaluation job,
        Then the API returns 403 Forbidden.
        """
        pytest.skip("Requires unprivileged user token — test deferred to RBAC setup")

    @pytest.mark.tier2
    def test_get_nonexistent_job_returns_404(
        self,
        evalhub_base_url: str,
        current_client_token: str,
        eval_resource_flavor: ResourceFlavor,
        eval_cluster_queue: ClusterQueue,
        eval_local_queue: LocalQueue,
        eval_test_namespace: Namespace,
    ) -> None:
        """TC-NEG-005: Verify GET non-existent job returns 404.

        Given no job exists with a specific ID,
        When a GET request is made for that job ID,
        Then the API returns 404 Not Found.
        """
        fake_job_id = "00000000-0000-0000-0000-000000000000"
        status_code, body = get_eval_job(base_url=evalhub_base_url, token=current_client_token, job_id=fake_job_id)

        assert status_code == 404, f"Expected 404, got {status_code}: {body}"
