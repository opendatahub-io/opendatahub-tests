import pytest
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import DynamicApiError
from ocp_resources.resource import ResourceEditor

from tests.model_serving.maas_billing.multitenancy.aigateway.utils import (
    AIGATEWAY_CREATED_ANNOTATION,
    MUTATED_TENANT_NAMESPACE_NAME,
    AIGatewayPreexistingNamespaceContext,
    AIGatewayTestContext,
    verify_aigateway_bootstrap_children_removed,
    verify_tenant_namespace_preserved,
)


@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest", "aigateway_infra_namespace")
class TestAIGatewayCleanup:
    """Tier1/tier2/tier3 tests for AIGateway deletion and immutability."""

    @pytest.mark.tier1
    def test_aigateway_deletion_cleans_up_children(
        self,
        admin_client: DynamicClient,
        ready_aigateway_with_cleanup_on_delete: AIGatewayTestContext,
    ) -> None:
        """Verify delete with cleanupOnDelete=true removes owned children."""
        test_context = ready_aigateway_with_cleanup_on_delete
        aigateway = test_context["aigateway"]
        aigateway.delete()
        aigateway.wait_deleted(timeout=300)
        verify_aigateway_bootstrap_children_removed(
            admin_client=admin_client,
            test_context=test_context,
        )

    @pytest.mark.tier1
    def test_aigateway_deletion_preserves_namespace_when_cleanup_disabled(
        self,
        admin_client: DynamicClient,
        ready_aigateway_without_cleanup_on_delete: AIGatewayTestContext,
    ) -> None:
        """Verify delete with cleanupOnDelete=false keeps the tenant namespace."""
        test_context = ready_aigateway_without_cleanup_on_delete
        aigateway = test_context["aigateway"]
        aigateway.delete()
        aigateway.wait_deleted(timeout=300)
        verify_tenant_namespace_preserved(
            admin_client=admin_client,
            tenant_namespace_name=test_context["tenant_namespace_name"],
        )

    @pytest.mark.tier3
    def test_aigateway_tenant_namespace_name_immutable(
        self,
        aigateway_for_test: AIGatewayTestContext,
    ) -> None:
        """Verify tenantNamespace.name cannot be patched after create."""
        aigateway = aigateway_for_test["aigateway"]
        with pytest.raises(DynamicApiError):
            ResourceEditor(
                patches={
                    aigateway: {
                        "spec": {
                            "tenantNamespace": {
                                "name": MUTATED_TENANT_NAMESPACE_NAME,
                                "create": True,
                                "cleanupOnDelete": True,
                            }
                        }
                    }
                }
            ).update()

    @pytest.mark.tier2
    def test_aigateway_deletion_preserves_preexisting_unmanaged_namespace(
        self,
        admin_client: DynamicClient,
        aigateway_on_labeled_preexisting_namespace: AIGatewayPreexistingNamespaceContext,
    ) -> None:
        """Verify delete does not remove a pre-existing unmanaged tenant namespace."""
        test_context = aigateway_on_labeled_preexisting_namespace
        aigateway = test_context["aigateway"]
        tenant_namespace = test_context["tenant_namespace"]
        assert AIGATEWAY_CREATED_ANNOTATION not in (tenant_namespace.instance.metadata.annotations or {}), (
            "Pre-existing namespace must not be marked created-by-aigateway"
        )
        aigateway.delete()
        aigateway.wait_deleted(timeout=300)
        verify_tenant_namespace_preserved(
            admin_client=admin_client,
            tenant_namespace_name=test_context["tenant_namespace_name"],
        )
