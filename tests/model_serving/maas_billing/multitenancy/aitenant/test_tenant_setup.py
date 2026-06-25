import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.custom_resource_definition import CustomResourceDefinition

from tests.model_serving.maas_billing.multitenancy.aitenant.utils import (
    AITENANT_CRD_NAME,
    AITenantTestContext,
    verify_aitenant_bootstrap_children,
    verify_aitenant_ready,
    verify_default_maas_tenant_unaffected,
)

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest")
class TestAITenantTenantSetup:
    @pytest.mark.smoke
    def test_aitenant_crd_exists(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Verify the AITenant CRD is registered in the cluster."""
        crd = CustomResourceDefinition(
            client=admin_client,
            name=AITENANT_CRD_NAME,
            ensure_exists=True,
        )
        assert crd.exists, f"AITenant CRD '{AITENANT_CRD_NAME}' not found in the cluster"
        LOGGER.info(f"AITenant CRD '{AITENANT_CRD_NAME}' exists")

    @pytest.mark.smoke
    def test_aitenant_bootstrap_creates_tenant_environment(
        self,
        admin_client: DynamicClient,
        aitenant_for_test: AITenantTestContext,
    ) -> None:
        """Verify AITenant bootstrap creates tenant namespace, Gateway, and default-tenant."""
        verify_aitenant_ready(aitenant=aitenant_for_test["aitenant"])
        verify_aitenant_bootstrap_children(
            admin_client=admin_client,
            test_context=aitenant_for_test,
        )

    @pytest.mark.smoke
    def test_aitenant_bootstrap_does_not_break_default_tenant(
        self,
        admin_client: DynamicClient,
        aitenant_for_test: AITenantTestContext,
    ) -> None:
        """Verify bootstrapping a tenant via AITenant does not affect default-tenant readiness."""
        verify_aitenant_ready(aitenant=aitenant_for_test["aitenant"])
        verify_default_maas_tenant_unaffected(admin_client=admin_client)
