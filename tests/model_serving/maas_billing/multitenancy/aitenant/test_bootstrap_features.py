import pytest
import structlog
from kubernetes.dynamic import DynamicClient

from tests.model_serving.maas_billing.multitenancy.aitenant.utils import (
    AIGATEWAY_BOOTSTRAPPED_TENANT_NAME,
    AITENANT_TEST_OIDC_SPEC,
    AITenantTestContext,
    verify_aitenant_bootstrap_children,
    verify_aitenant_ready,
    verify_bootstrapped_tenant_oidc,
)
from utilities.resources.aitenant import AITenant

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest", "aitenant_infra_namespace")
class TestAITenantBootstrapFeatures:
    """Check AITenant bootstrap readiness and OIDC mirroring."""

    @pytest.mark.tier1
    def test_aitenant_bootstrap_children_stay_ready(
        self,
        admin_client: DynamicClient,
        aitenant_for_test: AITenantTestContext,
    ) -> None:
        """Verify a bootstrapped AITenant reports Ready and creates expected child resources."""
        verify_aitenant_ready(aitenant=aitenant_for_test["aitenant"])
        verify_aitenant_bootstrap_children(
            admin_client=admin_client,
            test_context=aitenant_for_test,
        )

    @pytest.mark.tier1
    def test_aitenant_stays_ready_after_refetch(
        self,
        admin_client: DynamicClient,
        aitenant_for_test: AITenantTestContext,
    ) -> None:
        """Verify a bootstrapped AITenant stays ready after reconcile is re-checked from a fresh client."""
        aitenant = aitenant_for_test["aitenant"]
        refreshed_aitenant = AITenant(
            client=admin_client,
            name=aitenant.name,
            namespace=aitenant.namespace,
            wait_for_resource=False,
        )
        verify_aitenant_ready(aitenant=refreshed_aitenant)
        verify_aitenant_bootstrap_children(
            admin_client=admin_client,
            test_context=aitenant_for_test,
        )

    @pytest.mark.tier2
    def test_aitenant_oidc_mirrored_to_bootstrapped_tenant(
        self,
        admin_client: DynamicClient,
        aitenant_with_oidc: AITenantTestContext,
    ) -> None:
        """Verify spec.oidc is mirrored to Tenant/default-tenant externalOIDC."""
        verify_bootstrapped_tenant_oidc(
            admin_client=admin_client,
            tenant_namespace_name=aitenant_with_oidc["tenant_namespace_name"],
            expected_oidc=AITENANT_TEST_OIDC_SPEC,
        )
        LOGGER.info(
            f"AITenant oidc mirrored to Tenant/{AIGATEWAY_BOOTSTRAPPED_TENANT_NAME} in "
            f"'{aitenant_with_oidc['tenant_namespace_name']}'"
        )
