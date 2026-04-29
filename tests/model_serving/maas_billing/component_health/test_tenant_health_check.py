import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.namespace import Namespace

from utilities.constants import ApiGroups
from utilities.resources.tenant import Tenant

LOGGER = structlog.get_logger(name=__name__)

TENANT_CRD_NAME = f"tenants.{ApiGroups.MAAS_IO}"
TENANT_NAME = "default-tenant"


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
        admin_client: DynamicClient,
        maas_subscription_namespace: Namespace,
    ) -> None:
        """Verify default-tenant CR exists in the MaaS namespace."""
        tenant = Tenant(
            client=admin_client,
            name=TENANT_NAME,
            namespace=maas_subscription_namespace.name,
        )
        assert tenant.exists, f"Tenant '{TENANT_NAME}' not found in namespace '{maas_subscription_namespace.name}'"
        LOGGER.info(f"Tenant '{TENANT_NAME}' exists in namespace '{maas_subscription_namespace.name}'")

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
        admin_client: DynamicClient,
        maas_subscription_namespace: Namespace,
        condition_type: str,
        expected_status: str,
    ) -> None:
        """Verify a specific Tenant condition has the expected status."""
        tenant = Tenant(
            client=admin_client,
            name=TENANT_NAME,
            namespace=maas_subscription_namespace.name,
        )
        assert tenant.exists, f"Tenant '{TENANT_NAME}' not found"

        conditions = tenant.instance.status.get("conditions", [])
        for condition in conditions:
            if condition.get("type") == condition_type:
                assert condition.get("status") == expected_status, (
                    f"Tenant condition '{condition_type}': expected status='{expected_status}', "
                    f"got '{condition.get('status')}', reason='{condition.get('reason')}', "
                    f"message='{condition.get('message')}'"
                )
                LOGGER.info(f"Tenant condition '{condition_type}' is '{expected_status}'")
                break
        else:
            pytest.fail(f"Tenant condition '{condition_type}' not found in status")
