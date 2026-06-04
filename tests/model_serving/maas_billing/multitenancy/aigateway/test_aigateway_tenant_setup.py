import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.custom_resource_definition import CustomResourceDefinition

from tests.model_serving.maas_billing.multitenancy.aigateway.utils import (
    AIGATEWAY_CRD_NAME,
    AIGatewayTestContext,
    verify_aigateway_bootstrap_children,
    verify_aigateway_ready,
    verify_default_maas_tenant_unaffected,
)

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest")
class TestAIGatewayTenantSetup:
    @pytest.mark.smoke
    def test_aigateway_crd_exists(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Verify the AIGateway CRD is registered in the cluster."""
        crd = CustomResourceDefinition(
            client=admin_client,
            name=AIGATEWAY_CRD_NAME,
            ensure_exists=True,
        )
        assert crd.exists, f"AIGateway CRD '{AIGATEWAY_CRD_NAME}' not found in the cluster"
        LOGGER.info(f"AIGateway CRD '{AIGATEWAY_CRD_NAME}' exists")

    @pytest.mark.smoke
    def test_aigateway_bootstrap_creates_tenant_environment(
        self,
        admin_client: DynamicClient,
        aigateway_for_test: AIGatewayTestContext,
    ) -> None:
        """Verify AIGateway bootstrap creates tenant namespace, Gateway, and default-tenant."""
        verify_aigateway_ready(aigateway=aigateway_for_test["aigateway"])
        verify_aigateway_bootstrap_children(
            admin_client=admin_client,
            test_context=aigateway_for_test,
        )

    @pytest.mark.smoke
    def test_aigateway_bootstrap_does_not_break_default_tenant(
        self,
        admin_client: DynamicClient,
        aigateway_for_test: AIGatewayTestContext,
    ) -> None:
        """Verify bootstrapping a tenant via AIGateway does not affect default-tenant readiness."""
        verify_aigateway_ready(aigateway=aigateway_for_test["aigateway"])
        verify_default_maas_tenant_unaffected(admin_client=admin_client)
