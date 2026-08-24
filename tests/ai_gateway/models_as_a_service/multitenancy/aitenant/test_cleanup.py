import pytest
from kubernetes.dynamic import DynamicClient

from tests.ai_gateway.models_as_a_service.multitenancy.aitenant.utils import (
    AITenantPreexistingNamespaceContext,
    AITenantTestContext,
    delete_aitenant_and_wait,
    verify_aitenant_bootstrap_children_removed,
    verify_gateway_access_label_removed_after_aitenant_delete,
    verify_tenant_namespace_aitenant_metadata_stripped,
    verify_tenant_namespace_preserved,
)


@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest", "aitenant_infra_namespace")
class TestAITenantCleanup:
    """Tier1/tier2 tests for AITenant deletion behavior."""

    @pytest.mark.tier1
    def test_aitenant_deletion_removes_gateway_access_label(
        self,
        admin_client: DynamicClient,
        ready_aitenant_for_deletion: AITenantTestContext,
    ) -> None:
        """Given a Ready AITenant with gateway-access on the tenant namespace,
        when the AITenant is deleted,
        then maas.opendatahub.io/gateway-access is removed from the preserved namespace.
        """
        verify_gateway_access_label_removed_after_aitenant_delete(
            admin_client=admin_client,
            tenant_namespace_name=ready_aitenant_for_deletion["tenant_namespace_name"],
            aitenant=ready_aitenant_for_deletion["aitenant"],
        )

    @pytest.mark.tier1
    def test_aitenant_deletion_cleans_up_children_and_preserves_namespace(
        self,
        admin_client: DynamicClient,
        aitenant_infra_namespace: str,
        ready_aitenant_for_deletion: AITenantTestContext,
    ) -> None:
        """Verify delete removes controller-owned children and preserves the tenant namespace."""
        test_context = ready_aitenant_for_deletion
        delete_aitenant_and_wait(aitenant=test_context["aitenant"])
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
    def test_aitenant_applies_gateway_access_label_on_preexisting_namespace(
        self,
        admin_client: DynamicClient,
        aitenant_on_preexisting_derived_tenant_namespace: AITenantPreexistingNamespaceContext,
    ) -> None:
        """Given a pre-existing derived tenant namespace without gateway-access,
        when AITenant adopts it and becomes Ready,
        then maas.opendatahub.io/gateway-access=true is present,
        and when the AITenant is deleted the label is removed.
        """
        verify_gateway_access_label_removed_after_aitenant_delete(
            admin_client=admin_client,
            tenant_namespace_name=aitenant_on_preexisting_derived_tenant_namespace["tenant_namespace_name"],
            aitenant=aitenant_on_preexisting_derived_tenant_namespace["aitenant"],
        )

    @pytest.mark.tier2
    def test_aitenant_deletion_preserves_preexisting_tenant_namespace(
        self,
        admin_client: DynamicClient,
        aitenant_on_preexisting_derived_tenant_namespace: AITenantPreexistingNamespaceContext,
    ) -> None:
        """Verify delete preserves a pre-existing derived tenant namespace and strips AITenant metadata."""
        test_context = aitenant_on_preexisting_derived_tenant_namespace
        delete_aitenant_and_wait(aitenant=test_context["aitenant"])
        verify_tenant_namespace_preserved(
            admin_client=admin_client,
            tenant_namespace_name=test_context["tenant_namespace_name"],
        )
        verify_tenant_namespace_aitenant_metadata_stripped(
            admin_client=admin_client,
            tenant_namespace_name=test_context["tenant_namespace_name"],
        )
