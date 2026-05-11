"""Priority-based scheduling tests for EvalHub Kueue integration (TC-PRIO-001, TC-PRIO-002)."""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace

from tests.eval_hub.evalhub_kueue_integration.constants import (
    CLUSTER_QUEUE_NAME,
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
    wait_for_job_state,
)
from utilities.kueue_utils import ClusterQueue, LocalQueue, ResourceFlavor, WorkloadPriorityClass

pytestmark = [
    pytest.mark.kueue,
    pytest.mark.tier2,
]


class TestDefaultPriority:
    """TC-PRIO-001: Verify default priority 0 is assigned when not specified."""

    NAMESPACE_NAME = "evalhub-prio-default"

    @pytest.mark.parametrize(
        "eval_test_namespace, eval_resource_flavor, eval_cluster_queue, eval_local_queue",
        [
            pytest.param(
                {"name": "evalhub-prio-default"},
                {"name": RESOURCE_FLAVOR_NAME},
                {
                    "name": CLUSTER_QUEUE_NAME,
                    "resource_flavor_name": RESOURCE_FLAVOR_NAME,
                    "cpu_quota": 2,
                    "memory_quota": "8Gi",
                },
                {"name": LOCAL_QUEUE_NAME, "cluster_queue": CLUSTER_QUEUE_NAME},
                id="test_default_priority",
            )
        ],
        indirect=True,
    )
    def test_default_priority_assigned(
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
        """Verify that jobs submitted without priority get default priority 0.

        Given a configured Kueue queue,
        When an evaluation job is submitted without specifying priority,
        Then the job has default priority 0.
        """
        _, body = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-prio-001-default",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=LOCAL_QUEUE_NAME,
            tenant=eval_test_namespace.name,
        )
        job_id = body["resource"]["id"]

        _, status_body = get_eval_job(base_url=evalhub_base_url, token=current_client_token, job_id=job_id, tenant=eval_test_namespace.name)
        priority = status_body.get("priority", 0)
        assert priority == 0, f"Expected default priority 0, got {priority}"

        delete_eval_job(base_url=evalhub_base_url, token=current_client_token, job_id=job_id, hard_delete=True, tenant=eval_test_namespace.name)


class TestPriorityOrdering:
    """TC-PRIO-002: Verify higher priority job is admitted before lower priority."""

    @pytest.mark.parametrize(
        "eval_test_namespace, eval_resource_flavor, eval_cluster_queue, eval_local_queue, eval_workload_priority_class",
        [
            pytest.param(
                {"name": "evalhub-prio-order"},
                {"name": RESOURCE_FLAVOR_NAME},
                {
                    "name": CLUSTER_QUEUE_NAME,
                    "resource_flavor_name": RESOURCE_FLAVOR_NAME,
                    "cpu_quota": SMALL_CPU_QUOTA,
                    "memory_quota": SMALL_MEMORY_QUOTA,
                },
                {"name": LOCAL_QUEUE_NAME, "cluster_queue": CLUSTER_QUEUE_NAME},
                {"name": "evalhub-test-high-priority", "value": 1000},
                id="test_priority_ordering",
            )
        ],
        indirect=True,
    )
    def test_higher_priority_admitted_first(
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
        eval_workload_priority_class: WorkloadPriorityClass,
    ) -> None:
        """Verify higher priority job gets admitted before lower priority when quota is available.

        Given a ClusterQueue with limited quota and a running job,
        When a low-priority and a high-priority job are both pending,
        Then the high-priority job is admitted first when quota frees up.
        """
        _, body_blocker = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-prio-002-blocker",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=LOCAL_QUEUE_NAME,
            tenant=eval_test_namespace.name,
        )
        blocker_id = body_blocker["resource"]["id"]
        wait_for_job_running_or_completed(base_url=evalhub_base_url, token=current_client_token, job_id=blocker_id, tenant=eval_test_namespace.name)

        _, body_low = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-prio-002-low",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=LOCAL_QUEUE_NAME,
            priority=100,
            tenant=eval_test_namespace.name,
        )
        low_id = body_low["resource"]["id"]

        _, body_high = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-prio-002-high",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=LOCAL_QUEUE_NAME,
            priority=1000,
            tenant=eval_test_namespace.name,
        )
        high_id = body_high["resource"]["id"]

        wait_for_job_state(
            base_url=evalhub_base_url,
            token=current_client_token,
            job_id=blocker_id,
            target_state=EvalJobState.COMPLETED,
            tenant=eval_test_namespace.name,
        )

        high_result = wait_for_job_running_or_completed(
            base_url=evalhub_base_url, token=current_client_token, job_id=high_id,
            tenant=eval_test_namespace.name,
        )
        assert high_result, "High priority job should be admitted first"

        _, low_status_body = get_eval_job(base_url=evalhub_base_url, token=current_client_token, job_id=low_id, tenant=eval_test_namespace.name)
        low_state = low_status_body.get("status", {}).get("state")
        assert low_state == EvalJobState.PENDING, (
            f"Low priority job should stay {EvalJobState.PENDING} while high priority "
            f"(job_id={high_id}) is admitted first; got {low_state!r}"
        )

        for jid in (high_id, low_id, blocker_id):
            delete_eval_job(base_url=evalhub_base_url, token=current_client_token, job_id=jid, hard_delete=True, tenant=eval_test_namespace.name)
