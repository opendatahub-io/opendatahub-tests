import pytest

from tests.model_serving.maas_billing.multitenancy.aigateway.utils import (
    AIGATEWAY_TENANT_NAMESPACE_FAILED_REASON,
    AIGATEWAY_TENANT_NAMESPACE_MISSING_REASON,
    wait_until_aigateway_status,
)
from utilities.resources.aitenant import AITenant


@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest", "aigateway_infra_namespace")
class TestAIGatewayReconcileFailures:
    """Tier2 tests for AITenant reconcile failures."""

    @pytest.mark.tier2
    def test_aigateway_pending_when_create_disabled_and_namespace_missing(
        self,
        aigateway_pending_missing_tenant_namespace: AITenant,
    ) -> None:
        """Verify missing tenant namespace leaves AIGateway Pending."""
        wait_until_aigateway_status(
            aigateway=aigateway_pending_missing_tenant_namespace,
            phase="Pending",
            ready_reason=AIGATEWAY_TENANT_NAMESPACE_MISSING_REASON,
        )

    @pytest.mark.tier2
    def test_aigateway_rejects_namespace_owned_by_another_aigateway(
        self,
        aigateway_on_namespace_owned_by_other: AITenant,
    ) -> None:
        """Verify a namespace claimed by another AIGateway fails reconciliation."""
        wait_until_aigateway_status(
            aigateway=aigateway_on_namespace_owned_by_other,
            phase="Failed",
            ready_reason=AIGATEWAY_TENANT_NAMESPACE_FAILED_REASON,
        )
