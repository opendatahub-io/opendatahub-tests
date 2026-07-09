import pytest
import structlog
from kubernetes.dynamic import DynamicClient

from tests.model_serving.maas_billing.multitenancy.aitenant.utils import (
    AITENANT_TEST_RBAC_ADMINS,
    TEST_RBAC_GROUP_NAME,
    AITenantTestContext,
    verify_aitenant_controller_creates_admin_roles_only,
    verify_manual_aitenant_admin_role_bindings,
)

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest", "aitenant_infra_namespace")
class TestAITenantRbac:
    """Check AITenant controller admin Roles and manual RoleBinding provisioning."""

    @pytest.mark.tier1
    def test_aitenant_controller_creates_admin_roles_only(
        self,
        admin_client: DynamicClient,
        aitenant_infra_namespace: str,
        aitenant_for_test: AITenantTestContext,
    ) -> None:
        """Verify the controller creates tenant-admin Roles but does not create RoleBindings."""
        test_context = aitenant_for_test
        verify_aitenant_controller_creates_admin_roles_only(
            admin_client=admin_client,
            aitenant_name=test_context["aitenant_name"],
            tenant_namespace_name=test_context["tenant_namespace_name"],
            infra_namespace=aitenant_infra_namespace,
        )
        LOGGER.info(
            f"AITenant controller RBAC Roles verified without RoleBindings in "
            f"'{test_context['tenant_namespace_name']}' and '{aitenant_infra_namespace}'"
        )

    @pytest.mark.tier2
    def test_manual_aitenant_admin_role_bindings(
        self,
        admin_client: DynamicClient,
        aitenant_infra_namespace: str,
        aitenant_with_manual_admin_role_bindings: AITenantTestContext,
    ) -> None:
        """Verify manually created RoleBindings grant access via controller-provisioned Roles."""
        test_context = aitenant_with_manual_admin_role_bindings
        verify_manual_aitenant_admin_role_bindings(
            admin_client=admin_client,
            aitenant_name=test_context["aitenant_name"],
            tenant_namespace_name=test_context["tenant_namespace_name"],
            infra_namespace=aitenant_infra_namespace,
            expected_subjects=AITENANT_TEST_RBAC_ADMINS,
        )
        LOGGER.info(
            f"Manual AITenant admin RoleBindings verified for group '{TEST_RBAC_GROUP_NAME}' "
            f"in '{test_context['tenant_namespace_name']}' and '{aitenant_infra_namespace}'"
        )
