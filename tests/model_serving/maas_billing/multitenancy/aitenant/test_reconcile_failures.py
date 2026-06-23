import pytest

from tests.model_serving.maas_billing.multitenancy.aitenant.utils import (
    AITENANT_TENANT_NAMESPACE_FAILED_REASON,
    wait_until_aitenant_status,
)
from utilities.resources.aitenant import AITenant


@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest", "aitenant_infra_namespace")
class TestAITenantReconcileFailures:
    """Tier2 tests for AITenant reconcile failures."""

    @pytest.mark.tier2
    def test_aitenant_rejects_namespace_owned_by_another_aitenant(
        self,
        aitenant_on_namespace_owned_by_other: AITenant,
    ) -> None:
        """Verify a derived tenant namespace claimed by another AITenant fails reconciliation."""
        wait_until_aitenant_status(
            aitenant=aitenant_on_namespace_owned_by_other,
            phase="Failed",
            ready_reason=AITENANT_TENANT_NAMESPACE_FAILED_REASON,
        )
