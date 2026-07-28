import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.custom_resource_definition import CustomResourceDefinition

from utilities.constants import ApiGroups
from utilities.resources.maastenantconfig import MaasTenantConfig

LOGGER = structlog.get_logger(name=__name__)

MAAS_TENANT_CONFIG_CRD_NAME = f"maastenantconfigs.{ApiGroups.MAAS_IO}"


@pytest.mark.component_health
@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest")
class TestTenantHealthCheck:
    def test_maas_tenant_config_crd_exists(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Verify the MaasTenantConfig CRD is registered in the cluster."""
        crd = CustomResourceDefinition(
            client=admin_client,
            name=MAAS_TENANT_CONFIG_CRD_NAME,
            ensure_exists=True,
        )
        assert crd.exists, f"MaasTenantConfig CRD '{MAAS_TENANT_CONFIG_CRD_NAME}' not found in the cluster"
        LOGGER.info(f"MaasTenantConfig CRD '{MAAS_TENANT_CONFIG_CRD_NAME}' exists")

    @pytest.mark.parametrize(
        "condition_type, expected_status",
        [
            ("Ready", "True"),
            ("DependenciesAvailable", "True"),
            ("MaaSPrerequisitesAvailable", "True"),
            ("DeploymentsAvailable", "True"),
        ],
    )
    def test_maas_tenant_config_condition_healthy(
        self,
        default_maas_tenant_config: MaasTenantConfig,
        condition_type: str,
        expected_status: str,
    ) -> None:
        """Verify a specific MaasTenantConfig condition has the expected status."""
        default_maas_tenant_config.wait_for_condition(
            condition=condition_type,
            status=expected_status,
            timeout=120,
        )
        LOGGER.info(f"MaasTenantConfig condition '{condition_type}' is '{expected_status}'")
