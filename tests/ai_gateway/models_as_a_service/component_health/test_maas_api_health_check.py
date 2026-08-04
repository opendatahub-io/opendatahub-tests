import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment

from utilities.constants import DscComponents
from utilities.general import wait_for_pods_running

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.component_health
@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest")
class TestMaaSApiComponentHealth:
    def test_maas_management_state(
        self,
        dsc_resource: DataScienceCluster,
    ) -> None:
        """Verify aigateway and modelsAsAService managementState are MANAGED in DSC."""
        aigateway = dsc_resource.instance.spec.components[DscComponents.AIGATEWAY]
        assert aigateway.managementState == DscComponents.ManagementState.MANAGED
        assert aigateway.modelsAsAService.managementState == DscComponents.ManagementState.MANAGED

    def test_maas_condition_in_dsc(
        self,
        dsc_resource: DataScienceCluster,
    ) -> None:
        """Verify AIGatewayReady condition is True in DSC."""
        for condition in dsc_resource.instance.status.conditions:
            if condition.type == DscComponents.ConditionType.AIGATEWAY_READY:
                assert condition.status == "True"
                break
        else:
            pytest.fail(f"{DscComponents.ConditionType.AIGATEWAY_READY} condition not found in DSC")

    def test_maas_api_deployment_available(
        self,
        admin_client: DynamicClient,
        maas_api_infra_namespace: str,
    ) -> None:
        """Verify maas-api deployment Available=True in the infrastructure namespace."""
        maas_api_deployment = Deployment(
            client=admin_client,
            name="maas-api",
            namespace=maas_api_infra_namespace,
            ensure_exists=True,
        )
        maas_api_deployment.wait_for_condition(
            condition="Available",
            status="True",
            timeout=120,
        )

    def test_maas_api_pods_health(
        self,
        admin_client: DynamicClient,
        maas_api_infra_namespace: str,
    ) -> None:
        """Verify maas-api pods are Running/Ready in the infrastructure namespace."""
        LOGGER.info(f"Checking maas-api pods in namespace {maas_api_infra_namespace}")

        wait_for_pods_running(admin_client=admin_client, namespace_name=maas_api_infra_namespace)
