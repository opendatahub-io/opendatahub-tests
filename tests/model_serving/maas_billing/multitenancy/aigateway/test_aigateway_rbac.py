import pytest
import structlog
from kubernetes.dynamic import DynamicClient

from tests.model_serving.maas_billing.multitenancy.aigateway.utils import (
    TEST_RBAC_GROUP_NAME,
    aigateway_from_spec,
    build_aigateway_spec,
    deploy_and_verify_aigateway_ready,
    tenant_namespace_name_for_aigateway,
    verify_aigateway_rbac_admins_bindings,
    verify_aigateway_rbac_roles_without_admin_bindings,
)
from utilities.general import generate_random_name

LOGGER = structlog.get_logger(name=__name__)

TEST_RBAC_ADMINS = [{"kind": "Group", "name": TEST_RBAC_GROUP_NAME}]


@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest", "aigateway_infra_namespace")
class TestAIGatewayRbac:
    """Check admin RoleBindings are created when rbac.admins is set, and omitted when it is not."""

    @pytest.mark.tier1
    def test_aigateway_rbac_admins_creates_role_bindings(
        self,
        admin_client: DynamicClient,
        teardown_resources: bool,
        aigateway_infra_namespace: str,
    ) -> None:
        """Verify spec.rbac.admins creates tenant and infra RoleBindings for the admin group."""
        aigateway_name = f"e2e-aigw-rbac-{generate_random_name()}"
        tenant_namespace_name = tenant_namespace_name_for_aigateway(aigateway_name=aigateway_name)
        aigateway_spec = build_aigateway_spec(
            aigateway_name=aigateway_name,
            rbac_admins=TEST_RBAC_ADMINS,
        )
        with aigateway_from_spec(
            admin_client=admin_client,
            aigateway_name=aigateway_name,
            cr_namespace=aigateway_infra_namespace,
            aigateway_spec=aigateway_spec,
            teardown=teardown_resources,
        ) as aigateway:
            deploy_and_verify_aigateway_ready(aigateway=aigateway)
            verify_aigateway_rbac_admins_bindings(
                admin_client=admin_client,
                aigateway_name=aigateway_name,
                tenant_namespace_name=tenant_namespace_name,
                infra_namespace=aigateway_infra_namespace,
                expected_admins=TEST_RBAC_ADMINS,
            )
            LOGGER.info(
                f"AIGateway RBAC bindings verified for group '{TEST_RBAC_GROUP_NAME}' "
                f"in '{tenant_namespace_name}' and '{aigateway_infra_namespace}'"
            )

    @pytest.mark.tier2
    def test_aigateway_without_rbac_admins_omits_role_bindings(
        self,
        admin_client: DynamicClient,
        teardown_resources: bool,
        aigateway_infra_namespace: str,
    ) -> None:
        """Verify Roles exist but admin RoleBindings are omitted when spec.rbac.admins is unset."""
        aigateway_name = f"e2e-aigw-no-rbac-{generate_random_name()}"
        tenant_namespace_name = tenant_namespace_name_for_aigateway(aigateway_name=aigateway_name)
        aigateway_spec = build_aigateway_spec(aigateway_name=aigateway_name)
        with aigateway_from_spec(
            admin_client=admin_client,
            aigateway_name=aigateway_name,
            cr_namespace=aigateway_infra_namespace,
            aigateway_spec=aigateway_spec,
            teardown=teardown_resources,
        ) as aigateway:
            deploy_and_verify_aigateway_ready(aigateway=aigateway)
            verify_aigateway_rbac_roles_without_admin_bindings(
                admin_client=admin_client,
                aigateway_name=aigateway_name,
                tenant_namespace_name=tenant_namespace_name,
                infra_namespace=aigateway_infra_namespace,
            )
            LOGGER.info(
                f"AIGateway without rbac.admins omitted RoleBindings in "
                f"'{tenant_namespace_name}' and '{aigateway_infra_namespace}'"
            )
