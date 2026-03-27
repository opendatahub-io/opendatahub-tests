import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from pytest_testconfig import config as py_config

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
        """Verify modelsAsService managementState is MANAGED in DSC."""
        assert (
            dsc_resource.instance.spec.components[DscComponents.KSERVE].modelsAsService.managementState
            == DscComponents.ManagementState.MANAGED
        )

    def test_maas_condition_in_dsc(
        self,
        dsc_resource: DataScienceCluster,
    ) -> None:
        """Verify ModelsAsServiceReady condition is True in DSC."""
        for condition in dsc_resource.instance.status.conditions:
            if condition.type == "ModelsAsServiceReady":
                assert condition.status == "True"
                break
        else:
            pytest.fail("ModelsAsServiceReady condition not found in DSC")

    def test_maas_api_deployment_available(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Verify maas-api deployment Available=True."""
        applications_namespace = py_config["applications_namespace"]

        maas_api_deployment = Deployment(
            client=admin_client,
            name="maas-api",
            namespace=applications_namespace,
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
    ) -> None:
        """Verify maas-api pods are Running/Ready."""
        applications_namespace = py_config["applications_namespace"]
        LOGGER.info(f"Checking maas-api pods in namespace {applications_namespace}")

        wait_for_pods_running(admin_client=admin_client, namespace_name=applications_namespace)
