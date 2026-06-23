import pytest
from kubernetes.dynamic import DynamicClient

from tests.model_serving.maas_billing.multitenancy.aitenant.utils import (
    AITenantPreexistingNamespaceContext,
    AITenantTestContext,
    verify_aitenant_bootstrap_children_removed,
    verify_tenant_namespace_aitenant_metadata_stripped,
    verify_tenant_namespace_preserved,
)


@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest", "aitenant_infra_namespace")
class TestAITenantCleanup:
    """Tier1/tier2 tests for AITenant deletion behavior."""

    @pytest.mark.tier1
    def test_aitenant_deletion_cleans_up_children_and_preserves_namespace(
        self,
        admin_client: DynamicClient,
        aitenant_infra_namespace: str,
        ready_aitenant_for_deletion: AITenantTestContext,
    ) -> None:
        """Verify delete removes controller-owned children and preserves the tenant namespace."""
        test_context = ready_aitenant_for_deletion
        aitenant = test_context["aitenant"]
        aitenant.delete()
        aitenant.wait_deleted(timeout=300)
        verify_aitenant_bootstrap_children_removed(
            admin_client=admin_client,
            test_context=test_context,
            infra_namespace=aitenant_infra_namespace,
        )
        verify_tenant_namespace_preserved(
            admin_client=admin_client,
            tenant_namespace_name=test_context["tenant_namespace_name"],
        )
        verify_tenant_namespace_aitenant_metadata_stripped(
            admin_client=admin_client,
            tenant_namespace_name=test_context["tenant_namespace_name"],
        )

    @pytest.mark.tier2
    def test_aitenant_deletion_preserves_preexisting_tenant_namespace(
        self,
        admin_client: DynamicClient,
        aitenant_on_preexisting_derived_tenant_namespace: AITenantPreexistingNamespaceContext,
    ) -> None:
        """Verify delete preserves a pre-existing derived tenant namespace and strips AITenant metadata."""
        test_context = aitenant_on_preexisting_derived_tenant_namespace
        aitenant = test_context["aitenant"]
        aitenant.delete()
        aitenant.wait_deleted(timeout=300)
        verify_tenant_namespace_preserved(
            admin_client=admin_client,
            tenant_namespace_name=test_context["tenant_namespace_name"],
        )
        verify_tenant_namespace_aitenant_metadata_stripped(
            admin_client=admin_client,
            tenant_namespace_name=test_context["tenant_namespace_name"],
        )
