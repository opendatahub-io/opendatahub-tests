"""Upgrade tests for EvalHub Kueue integration (TC-UPGRADE-001, TC-UPGRADE-002)."""

import pytest
from ocp_resources.namespace import Namespace

from tests.eval_hub.evalhub_kueue_integration.constants import (
    CLUSTER_QUEUE_NAME,
    DEFAULT_CPU_QUOTA,
    DEFAULT_MEMORY_QUOTA,
    LOCAL_QUEUE_NAME,
    RESOURCE_FLAVOR_NAME,
)
from tests.eval_hub.evalhub_kueue_integration.utils import (
    delete_eval_job,
    get_health,
    submit_eval_job,
    wait_for_job_running_or_completed,
)
from utilities.kueue_utils import ClusterQueue, LocalQueue, ResourceFlavor

pytestmark = [
    pytest.mark.kueue,
    pytest.mark.tier2,
    pytest.mark.pre_upgrade,
    pytest.mark.post_upgrade,
]

NAMESPACE_NAME = "evalhub-upgrade-test"


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
            id="test_upgrade_scenarios",
        )
    ],
    indirect=True,
)
class TestUpgradeScenarios:
    """Upgrade tests validating EvalHub backwards compatibility with Kueue integration."""

    def test_backwards_compatibility(
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
        """TC-UPGRADE-001: Verify job submission works with and without Kueue across upgrade.

        Given EvalHub deployed before and after upgrade,
        When jobs are submitted both with and without queue specification,
        Then both modes work correctly on the current version.
        """
        status_no_queue, body_no_queue = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-upgrade-001-no-queue",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=None,
            tenant=eval_test_namespace.name,
        )
        assert status_no_queue == 202, f"Expected 202 without queue, got {status_no_queue}: {body_no_queue}"
        no_queue_id = body_no_queue["resource"]["id"]

        status_with_queue, body_with_queue = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-upgrade-001-with-queue",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=LOCAL_QUEUE_NAME,
            tenant=eval_test_namespace.name,
        )
        assert status_with_queue == 202, f"Expected 202 with queue, got {status_with_queue}: {body_with_queue}"
        with_queue_id = body_with_queue["resource"]["id"]

        result = wait_for_job_running_or_completed(
            base_url=evalhub_base_url, token=current_client_token, job_id=with_queue_id, tenant=eval_test_namespace.name
        )
        assert result, "Kueue-managed job should be admitted"

        delete_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            job_id=no_queue_id,
            hard_delete=True,
            tenant=eval_test_namespace.name,
        )
        delete_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            job_id=with_queue_id,
            hard_delete=True,
            tenant=eval_test_namespace.name,
        )

    def test_rollback_scenario(
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
        """TC-UPGRADE-002: Verify EvalHub functions when Kueue is removed (rollback scenario).

        Given EvalHub deployed with Kueue configured,
        When a job is submitted without queue specification,
        Then it runs normally regardless of Kueue presence,
        And the health endpoint returns healthy status.
        """
        status_code, body = submit_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            name="tc-upgrade-002-rollback",
            model_url=evalhub_model_url,
            model_name=evalhub_model_name,
            queue_name=None,
            tenant=eval_test_namespace.name,
        )
        assert status_code == 202, f"Expected 202, got {status_code}: {body}"
        job_id = body["resource"]["id"]

        health_status, health_body = get_health(base_url=evalhub_base_url, token=current_client_token)
        assert health_status == 200, f"Health check failed: {health_status} {health_body}"

        delete_eval_job(
            base_url=evalhub_base_url,
            token=current_client_token,
            job_id=job_id,
            hard_delete=True,
            tenant=eval_test_namespace.name,
        )
