import pytest
import structlog
from kubernetes.dynamic import DynamicClient

from tests.model_serving.maas_billing.multitenancy.aitenant.utils import (
    AITENANT_TEST_RBAC_ADMINS,
    TEST_RBAC_GROUP_NAME,
    AITenantTestContext,
    verify_aitenant_rbac_admins_bindings,
    verify_aitenant_rbac_roles_without_admin_bindings,
)

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest", "aitenant_infra_namespace")
class TestAITenantRbac:
    """Check admin RoleBindings are created when rbac.admins is set, and omitted when it is not."""

    @pytest.mark.tier1
    def test_aitenant_rbac_admins_creates_role_bindings(
        self,
        admin_client: DynamicClient,
        aitenant_infra_namespace: str,
        aitenant_with_rbac_admins: AITenantTestContext,
    ) -> None:
        """Verify spec.rbac.admins creates tenant and infra RoleBindings for the admin group."""
        test_context = aitenant_with_rbac_admins
        verify_aitenant_rbac_admins_bindings(
            admin_client=admin_client,
            aitenant_name=test_context["aitenant_name"],
            tenant_namespace_name=test_context["tenant_namespace_name"],
            infra_namespace=aitenant_infra_namespace,
            expected_admins=AITENANT_TEST_RBAC_ADMINS,
        )
        LOGGER.info(
            f"AITenant RBAC bindings verified for group '{TEST_RBAC_GROUP_NAME}' "
            f"in '{test_context['tenant_namespace_name']}' and '{aitenant_infra_namespace}'"
        )

    @pytest.mark.tier2
    def test_aitenant_without_rbac_admins_omits_role_bindings(
        self,
        admin_client: DynamicClient,
        aitenant_infra_namespace: str,
        aitenant_without_rbac_admins: AITenantTestContext,
    ) -> None:
        """Verify Roles exist but admin RoleBindings are omitted when spec.rbac.admins is unset."""
        test_context = aitenant_without_rbac_admins
        verify_aitenant_rbac_roles_without_admin_bindings(
            admin_client=admin_client,
            aitenant_name=test_context["aitenant_name"],
            tenant_namespace_name=test_context["tenant_namespace_name"],
            infra_namespace=aitenant_infra_namespace,
        )
        LOGGER.info(
            f"AITenant without rbac.admins omitted RoleBindings in "
            f"'{test_context['tenant_namespace_name']}' and '{aitenant_infra_namespace}'"
        )
