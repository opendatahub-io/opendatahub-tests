import pytest
import structlog
from kubernetes.dynamic import DynamicClient

from tests.model_serving.maas_billing.multitenancy.aigateway.utils import (
    AIGATEWAY_TEST_RBAC_ADMINS,
    TEST_RBAC_GROUP_NAME,
    AIGatewayTestContext,
    verify_aigateway_rbac_admins_bindings,
    verify_aigateway_rbac_roles_without_admin_bindings,
)

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest", "aigateway_infra_namespace")
class TestAIGatewayRbac:
    """Check admin RoleBindings are created when rbac.admins is set, and omitted when it is not."""

    @pytest.mark.tier1
    def test_aigateway_rbac_admins_creates_role_bindings(
        self,
        admin_client: DynamicClient,
        aigateway_infra_namespace: str,
        aigateway_with_rbac_admins: AIGatewayTestContext,
    ) -> None:
        """Verify spec.rbac.admins creates tenant and infra RoleBindings for the admin group."""
        test_context = aigateway_with_rbac_admins
        verify_aigateway_rbac_admins_bindings(
            admin_client=admin_client,
            aigateway_name=test_context["aigateway_name"],
            tenant_namespace_name=test_context["tenant_namespace_name"],
            infra_namespace=aigateway_infra_namespace,
            expected_admins=AIGATEWAY_TEST_RBAC_ADMINS,
        )
        LOGGER.info(
            f"AIGateway RBAC bindings verified for group '{TEST_RBAC_GROUP_NAME}' "
            f"in '{test_context['tenant_namespace_name']}' and '{aigateway_infra_namespace}'"
        )

    @pytest.mark.tier2
    def test_aigateway_without_rbac_admins_omits_role_bindings(
        self,
        admin_client: DynamicClient,
        aigateway_infra_namespace: str,
        aigateway_without_rbac_admins: AIGatewayTestContext,
    ) -> None:
        """Verify Roles exist but admin RoleBindings are omitted when spec.rbac.admins is unset."""
        test_context = aigateway_without_rbac_admins
        verify_aigateway_rbac_roles_without_admin_bindings(
            admin_client=admin_client,
            aigateway_name=test_context["aigateway_name"],
            tenant_namespace_name=test_context["tenant_namespace_name"],
            infra_namespace=aigateway_infra_namespace,
        )
        LOGGER.info(
            f"AIGateway without rbac.admins omitted RoleBindings in "
            f"'{test_context['tenant_namespace_name']}' and '{aigateway_infra_namespace}'"
        )
