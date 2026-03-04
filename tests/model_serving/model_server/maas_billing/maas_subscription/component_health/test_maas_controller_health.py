import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from pytest_testconfig import config as py_config
from simple_logger.logger import get_logger

from utilities.constants import ApiGroups
from utilities.general import wait_for_pods_running

LOGGER = get_logger(name=__name__)


@pytest.mark.smoke
@pytest.mark.usefixtures("maas_controller_enabled_latest")
class TestMaaSController:
    def test_maas_condition_in_dsc(
        self,
        dsc_resource: DataScienceCluster,
    ) -> None:
        """Verify ModelsAsServiceReady condition is True in DSC (MR-style loop + break + else fail)."""
        for condition in dsc_resource.instance.status.conditions:
            if condition.type == "ModelsAsServiceReady":
                assert condition.status == "True"
                break
        else:
            pytest.fail("ModelsAsServiceReady condition not found in DSC")

    def test_maas_controller_crds_exist(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Verify MaaS controller CRDs exist (no CustomResourceDefinition.get usage)."""
        expected_crd_names = (
            f"maasmodels.{ApiGroups.MAAS_IO}",
            f"maasauthpolicies.{ApiGroups.MAAS_IO}",
            f"maassubscriptions.{ApiGroups.MAAS_IO}",
        )

        for crd_name in expected_crd_names:
            crd_resource = CustomResourceDefinition(
                client=admin_client,
                name=crd_name,
                ensure_exists=True,
            )
            assert crd_resource.exists, f"Expected CRD to exist: {crd_name}"

    def test_maas_controller_deployment_available(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Verify maas-controller Deployment is Available."""
        applications_namespace = py_config["applications_namespace"]

        controller_deployment = Deployment(
            client=admin_client,
            name="maas-controller",
            namespace=applications_namespace,
            ensure_exists=True,
        )
        controller_deployment.wait_for_condition(
            condition="Available",
            status="True",
            timeout=600,
        )

    def test_maas_controller_pods_health(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Verify maas-controller pods are Running."""
        applications_namespace = py_config["applications_namespace"]
        LOGGER.info(f"Testing Pods in namespace {applications_namespace} for MaaS Controller health")
        wait_for_pods_running(admin_client=admin_client, namespace_name=applications_namespace)

    def test_maas_subscription_stack_ready_for_free_model(
        self,
        maas_model_tinyllama_free,
        maas_auth_policy_tinyllama_free,
        maas_subscription_tinyllama_free,
    ) -> None:
        """
        Verify the MaaS subscription flow objects are created and Ready.
        """
        LOGGER.info(f"Checking MaaSModel {maas_model_tinyllama_free.name} is Ready")
        maas_model_tinyllama_free.wait_for_condition(condition="Ready", status="True", timeout=300)

        LOGGER.info(f"Checking MaaSAuthPolicy {maas_auth_policy_tinyllama_free.name} is Ready")
        maas_auth_policy_tinyllama_free.wait_for_condition(condition="Ready", status="True", timeout=300)

        LOGGER.info(f"Checking MaaSSubscription {maas_subscription_tinyllama_free.name} is Ready")
        maas_subscription_tinyllama_free.wait_for_condition(condition="Ready", status="True", timeout=300)
