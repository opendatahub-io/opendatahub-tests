"""Multi-tenancy tests for EvalHub Kueue integration (TC-MULTI-001)."""

import pytest
from kubernetes.dynamic import DynamicClient

from tests.eval_hub.evalhub_kueue_integration.constants import (
    EVALHUB_TENANT_LABEL,
    KUEUE_MANAGED_LABEL,
    LOCAL_QUEUE_NAME,
    SMALL_CPU_QUOTA,
    SMALL_MEMORY_QUOTA,
)
from tests.eval_hub.evalhub_kueue_integration.utils import (
    delete_eval_job,
    submit_eval_job,
    wait_for_job_running_or_completed,
)
from utilities.infra import create_ns
from utilities.kueue_utils import (
    create_cluster_queue,
    create_local_queue,
    create_resource_flavor,
)

pytestmark = [
    pytest.mark.kueue,
    pytest.mark.tier3,
]

TEAM_A_NS = "evalhub-team-a"
TEAM_B_NS = "evalhub-team-b"
TEAM_A_CQ = "team-a-cq"
TEAM_B_CQ = "team-b-cq"
MULTI_TENANCY_FLAVOR = "evalhub-multi-test-flavor"


class TestMultiTenancy:
    """TC-MULTI-001: Verify jobs in different namespaces use separate ClusterQueues."""

    def test_namespace_quota_isolation(
        self,
        admin_client: DynamicClient,
        evalhub_base_url: str,
        current_client_token: str,
        evalhub_model_url: str,
        evalhub_model_name: str,
        ensure_kueue_available: None,
    ) -> None:
        """Verify that teams in different namespaces have isolated quotas.

        Given two namespaces with separate ClusterQueues and equal quotas,
        When jobs are submitted from both namespaces,
        Then each namespace's quota is enforced independently.
        """
        with (
            create_resource_flavor(client=admin_client, name=MULTI_TENANCY_FLAVOR),
            create_ns(
                name=TEAM_A_NS,
                admin_client=admin_client,
                labels={KUEUE_MANAGED_LABEL: "true", EVALHUB_TENANT_LABEL: "true", "team": "team-a"},
            ),
            create_ns(
                name=TEAM_B_NS,
                admin_client=admin_client,
                labels={KUEUE_MANAGED_LABEL: "true", EVALHUB_TENANT_LABEL: "true", "team": "team-b"},
            ),
            create_cluster_queue(
                client=admin_client,
                name=TEAM_A_CQ,
                resource_groups=[
                    {
                        "coveredResources": ["cpu", "memory"],
                        "flavors": [
                            {
                                "name": MULTI_TENANCY_FLAVOR,
                                "resources": [
                                    {"name": "cpu", "nominalQuota": SMALL_CPU_QUOTA},
                                    {"name": "memory", "nominalQuota": SMALL_MEMORY_QUOTA},
                                ],
                            }
                        ],
                    }
                ],
                namespace_selector={"matchLabels": {"team": "team-a"}},
            ),
            create_cluster_queue(
                client=admin_client,
                name=TEAM_B_CQ,
                resource_groups=[
                    {
                        "coveredResources": ["cpu", "memory"],
                        "flavors": [
                            {
                                "name": MULTI_TENANCY_FLAVOR,
                                "resources": [
                                    {"name": "cpu", "nominalQuota": SMALL_CPU_QUOTA},
                                    {"name": "memory", "nominalQuota": SMALL_MEMORY_QUOTA},
                                ],
                            }
                        ],
                    }
                ],
                namespace_selector={"matchLabels": {"team": "team-b"}},
            ),
            create_local_queue(
                client=admin_client, name=LOCAL_QUEUE_NAME, cluster_queue=TEAM_A_CQ, namespace=TEAM_A_NS
            ),
            create_local_queue(
                client=admin_client, name=LOCAL_QUEUE_NAME, cluster_queue=TEAM_B_CQ, namespace=TEAM_B_NS
            ),
        ):
            _, body_a = submit_eval_job(
                base_url=evalhub_base_url,
                token=current_client_token,
                name="tc-multi-001-team-a",
                model_url=evalhub_model_url,
                model_name=evalhub_model_name,
                queue_name=LOCAL_QUEUE_NAME,
                tenant="team-a",
            )
            job_a_id = body_a["resource"]["id"]

            _, body_b = submit_eval_job(
                base_url=evalhub_base_url,
                token=current_client_token,
                name="tc-multi-001-team-b",
                model_url=evalhub_model_url,
                model_name=evalhub_model_name,
                queue_name=LOCAL_QUEUE_NAME,
                tenant="team-b",
            )
            job_b_id = body_b["resource"]["id"]

            result_a = wait_for_job_running_or_completed(
                base_url=evalhub_base_url, token=current_client_token, job_id=job_a_id, tenant="team-a"
            )
            result_b = wait_for_job_running_or_completed(
                base_url=evalhub_base_url, token=current_client_token, job_id=job_b_id, tenant="team-b"
            )
            assert result_a, "Team A job should be admitted (independent quota)"
            assert result_b, "Team B job should be admitted (independent quota)"

            delete_eval_job(
                base_url=evalhub_base_url,
                token=current_client_token,
                job_id=job_a_id,
                hard_delete=True,
                tenant="team-a",
            )
            delete_eval_job(
                base_url=evalhub_base_url,
                token=current_client_token,
                job_id=job_b_id,
                hard_delete=True,
                tenant="team-b",
            )
