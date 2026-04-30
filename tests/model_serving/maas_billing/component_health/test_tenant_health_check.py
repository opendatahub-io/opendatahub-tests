import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.custom_resource_definition import CustomResourceDefinition

from utilities.constants import ApiGroups
from utilities.resources.tenant import Tenant

LOGGER = structlog.get_logger(name=__name__)

TENANT_CRD_NAME = f"tenants.{ApiGroups.MAAS_IO}"


@pytest.mark.component_health
@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest")
class TestTenantHealthCheck:
    def test_tenant_crd_exists(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Verify the Tenant CRD is registered in the cluster."""
        crd = CustomResourceDefinition(
            client=admin_client,
            name=TENANT_CRD_NAME,
            ensure_exists=True,
        )
        assert crd.exists, f"Tenant CRD '{TENANT_CRD_NAME}' not found in the cluster"
        LOGGER.info(f"Tenant CRD '{TENANT_CRD_NAME}' exists")

    def test_default_tenant_exists(
        self,
        default_tenant: Tenant,
    ) -> None:
        """Verify default-tenant CR exists in the MaaS namespace."""
        LOGGER.info(f"Tenant '{default_tenant.name}' exists in namespace '{default_tenant.namespace}'")

    @pytest.mark.parametrize(
        "condition_type, expected_status",
        [
            ("Ready", "True"),
            ("DependenciesAvailable", "True"),
            ("MaaSPrerequisitesAvailable", "True"),
            ("DeploymentsAvailable", "True"),
        ],
    )
    def test_tenant_condition_healthy(
        self,
        default_tenant: Tenant,
        condition_type: str,
        expected_status: str,
    ) -> None:
        """Verify a specific Tenant condition has the expected status."""
        default_tenant.wait_for_condition(
            condition=condition_type,
            status=expected_status,
            timeout=120,
        )
        LOGGER.info(f"Tenant condition '{condition_type}' is '{expected_status}'")
