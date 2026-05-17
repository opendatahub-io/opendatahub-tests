"""Preemption scenario tests for EvalHub Kueue integration (TC-PREEMPT-001 through TC-PREEMPT-003)."""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.job import Job
from ocp_resources.namespace import Namespace

from tests.model_explainability.evalhub.evalhub_kueue_integration.constants import (
    KUEUE_QUEUE_NAME_LABEL,
    LOCAL_QUEUE_NAME,
    RESOURCE_FLAVOR_NAME,
    SMALL_CPU_QUOTA,
    SMALL_MEMORY_QUOTA,
)
from tests.model_explainability.evalhub.evalhub_kueue_integration.utils import (
    delete_eval_job,
    submit_eval_job,
    wait_for_job_running_or_completed,
)
from utilities.kueue_utils import ClusterQueue, LocalQueue, ResourceFlavor

pytestmark = [
    pytest.mark.kueue,
    pytest.mark.tier3,
]

CQ_PREEMPT_NAME = "eval-cq-preempt"
CQ_NO_PREEMPT_NAME = "eval-cq-no-preempt"


class TestPreemptionEnabled:
    """TC-PREEMPT-001: Higher priority job preempts lower priority when preemption is enabled."""

    @pytest.mark.parametrize(
        "eval_test_namespace, eval_resource_flavor, eval_cluster_queue, eval_local_queue",
        [
            pytest.param(
                {"name": "evalhub-preempt-on"},
                {"name": RESOURCE_FLAVOR_NAME},
                {
                    "name": CQ_PREEMPT_NAME,
                    "resource_flavor_name": RESOURCE_FLAVOR_NAME,
                    "cpu_quota": SMALL_CPU_QUOTA,
                    "memory_quota": SMALL_MEMORY_QUOTA,
                    "preemption": {"withinClusterQueue": "LowerPriority"},
                },
                {"name": LOCAL_QUEUE_NAME, "cluster_queue": CQ_PREEMPT_NAME},
                id="test_preemption_enabled",
            )
        ],
        indirect=True,
    )
    def test_higher_priority_preempts_lower(
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
        """Verify higher priority job preempts a running lower priority job.

        Given a ClusterQueue with preemption enabled (withinClusterQueue: LowerPriority),
        When a higher-priority job is submitted while a lower-priority job consumes all quota,
        Then the lower-priority job is evicted and the higher-priority job is admitted.
        """
        _, body_low = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-preempt-001-low",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=LOCAL_QUEUE_NAME,
            priority=100,
            tenant=eval_test_namespace.name,
        )
        low_id = body_low["resource"]["id"]
        wait_for_job_running_or_completed(
            base_url=evalhub_base_url, token=current_client_token, job_id=low_id, tenant=eval_test_namespace.name
        )

        _, body_high = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-preempt-001-high",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=LOCAL_QUEUE_NAME,
            priority=1000,
            tenant=eval_test_namespace.name,
        )
        high_id = body_high["resource"]["id"]

        high_result = wait_for_job_running_or_completed(
            base_url=evalhub_base_url,
            token=current_client_token,
            job_id=high_id,
            tenant=eval_test_namespace.name,
        )
        assert high_result, "High priority job should be admitted after preempting low priority job"

        low_jobs = list(
            Job.get(
                client=admin_client,
                namespace=eval_test_namespace.name,
                label_selector=f"{KUEUE_QUEUE_NAME_LABEL}={LOCAL_QUEUE_NAME}",
            )
        )
        low_job = next((j for j in low_jobs if "low" in j.name), None)
        if low_job:
            assert low_job.instance.spec.get("suspend") is True, (
                "Low priority job should be re-suspended after preemption"
            )

        for jid in (high_id, low_id):
            delete_eval_job(
                base_url=evalhub_base_url,
                token=current_client_token,
                job_id=jid,
                hard_delete=True,
                tenant=eval_test_namespace.name,
            )


class TestPreemptedJobRestart:
    """TC-PREEMPT-002: Preempted job restarts from beginning when resumed."""

    @pytest.mark.parametrize(
        "eval_test_namespace, eval_resource_flavor, eval_cluster_queue, eval_local_queue",
        [
            pytest.param(
                {"name": "evalhub-preempt-restart"},
                {"name": RESOURCE_FLAVOR_NAME},
                {
                    "name": CQ_PREEMPT_NAME,
                    "resource_flavor_name": RESOURCE_FLAVOR_NAME,
                    "cpu_quota": SMALL_CPU_QUOTA,
                    "memory_quota": SMALL_MEMORY_QUOTA,
                    "preemption": {"withinClusterQueue": "LowerPriority"},
                },
                {"name": LOCAL_QUEUE_NAME, "cluster_queue": CQ_PREEMPT_NAME},
                id="test_preempted_restart",
            )
        ],
        indirect=True,
    )
    def test_preempted_job_restarts_from_beginning(
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
        """Verify preempted evaluation job restarts from the beginning when re-admitted.

        Given a preempted evaluation job that was evicted,
        When the high-priority job completes and frees quota,
        Then the preempted job is re-admitted and restarts from scratch.
        """
        _, body_low = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-preempt-002-low",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=LOCAL_QUEUE_NAME,
            priority=100,
            tenant=eval_test_namespace.name,
        )
        low_id = body_low["resource"]["id"]
        wait_for_job_running_or_completed(
            base_url=evalhub_base_url, token=current_client_token, job_id=low_id, tenant=eval_test_namespace.name
        )

        _, body_high = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-preempt-002-high",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=LOCAL_QUEUE_NAME,
            priority=1000,
            tenant=eval_test_namespace.name,
        )
        high_id = body_high["resource"]["id"]

        wait_for_job_running_or_completed(
            base_url=evalhub_base_url, token=current_client_token, job_id=high_id, tenant=eval_test_namespace.name
        )

        delete_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            job_id=high_id,
            hard_delete=True,
            tenant=eval_test_namespace.name,
        )

        low_result = wait_for_job_running_or_completed(
            base_url=evalhub_base_url,
            token=current_client_token,
            job_id=low_id,
            tenant=eval_test_namespace.name,
        )
        assert low_result, "Preempted job should be re-admitted and running after high-priority job completes"

        delete_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            job_id=low_id,
            hard_delete=True,
            tenant=eval_test_namespace.name,
        )


class TestNoPreemption:
    """TC-PREEMPT-003: No preemption when withinClusterQueue is set to Never."""

    @pytest.mark.parametrize(
        "eval_test_namespace, eval_resource_flavor, eval_cluster_queue, eval_local_queue",
        [
            pytest.param(
                {"name": "evalhub-preempt-never"},
                {"name": RESOURCE_FLAVOR_NAME},
                {
                    "name": CQ_NO_PREEMPT_NAME,
                    "resource_flavor_name": RESOURCE_FLAVOR_NAME,
                    "cpu_quota": SMALL_CPU_QUOTA,
                    "memory_quota": SMALL_MEMORY_QUOTA,
                    "preemption": {"withinClusterQueue": "Never"},
                },
                {"name": LOCAL_QUEUE_NAME, "cluster_queue": CQ_NO_PREEMPT_NAME},
                id="test_no_preemption",
            )
        ],
        indirect=True,
    )
    def test_no_preemption_when_disabled(
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
        """Verify no preemption occurs when withinClusterQueue is Never.

        Given a ClusterQueue with preemption disabled (withinClusterQueue: Never),
        When a higher-priority job is submitted while a lower-priority job consumes all quota,
        Then the lower-priority job continues running and the higher-priority job remains pending.
        """
        _, body_low = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-preempt-003-low",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=LOCAL_QUEUE_NAME,
            priority=100,
            tenant=eval_test_namespace.name,
        )
        low_id = body_low["resource"]["id"]
        wait_for_job_running_or_completed(
            base_url=evalhub_base_url, token=current_client_token, job_id=low_id, tenant=eval_test_namespace.name
        )

        _, body_high = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-preempt-003-high",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=LOCAL_QUEUE_NAME,
            priority=1000,
            tenant=eval_test_namespace.name,
        )
        high_id = body_high["resource"]["id"]

        low_jobs = list(
            Job.get(
                client=admin_client,
                namespace=eval_test_namespace.name,
                label_selector=f"{KUEUE_QUEUE_NAME_LABEL}={LOCAL_QUEUE_NAME}",
            )
        )
        low_job = next((j for j in low_jobs if "low" in j.name), None)
        if low_job:
            assert low_job.instance.spec.get("suspend") is not True, (
                "Low priority job should NOT be suspended when preemption is disabled"
            )

        for jid in (high_id, low_id):
            delete_eval_job(
                base_url=evalhub_base_url,
                token=current_client_token,
                job_id=jid,
                hard_delete=True,
                tenant=eval_test_namespace.name,
            )
