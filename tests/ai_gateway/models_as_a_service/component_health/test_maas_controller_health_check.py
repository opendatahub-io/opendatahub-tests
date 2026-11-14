from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from pytest_testconfig import config as py_config

from tests.ai_gateway.models_as_a_service.component_health.utils import (
    verify_ai_gateway_controller_deployment_available,
)
from tests.ai_gateway.models_as_a_service.utils import dsc_uses_aigateway_maas_schema
from utilities.constants import ApiGroups, DscComponents
from utilities.general import wait_for_pods_running

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.component_health
@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest")
class TestMaaSController:
    def test_maas_condition_in_dsc(
        self,
        dsc_resource: DataScienceCluster,
    ) -> None:
        """Verify AIGatewayReady condition is True in DSC (MR-style loop + break + else fail)."""
        for condition in dsc_resource.instance.status.conditions:
            if condition.type == DscComponents.ConditionType.AIGATEWAY_READY:
                assert condition.status == "True"
                break
        else:
            pytest.fail(f"{DscComponents.ConditionType.AIGATEWAY_READY} condition not found in DSC")

    def test_maas_controller_crds_exist(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Verify MaaS controller CRDs exist (no CustomResourceDefinition.get usage)."""
        expected_crd_names = (
            f"maasmodelrefs.{ApiGroups.MAAS_IO}",
            f"maasauthpolicies.{ApiGroups.MAAS_IO}",
            f"maassubscriptions.{ApiGroups.MAAS_IO}",
        )
        missing_crds = []
        for crd_name in expected_crd_names:
            crd_resource = CustomResourceDefinition(
                client=admin_client,
                name=crd_name,
                ensure_exists=True,
            )
            if not crd_resource.exists:
                missing_crds.append(crd_name)

        assert not missing_crds, f"Missing expected CRDs: {', '.join(missing_crds)}"

    def test_maas_controller_deployment_available(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Verify MaaS controller Deployment is Available using the schema-appropriate name.

        Given MaaS is managed, when checking the applications namespace,
        then the schema-appropriate controller deployment
        (ai-gateway-controller for the aigateway MaaS schema,
        maas-controller for the legacy schema) is Available.
        """
        if dsc_uses_aigateway_maas_schema(admin_client=admin_client):
            verify_ai_gateway_controller_deployment_available(admin_client=admin_client)
            return

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
            timeout=120,
        )

    def test_maas_controller_pods_health(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Verify all pods in the applications namespace are Running."""
        applications_namespace = py_config["applications_namespace"]
        LOGGER.info(f"Testing Pods in namespace {applications_namespace} for MaaS Controller health")
        wait_for_pods_running(admin_client=admin_client, namespace_name=applications_namespace)

    @pytest.mark.parametrize(
        "resource_fixture, kind_label",
        [
            ("maas_model_tinyllama_free", "MaaSModelRef"),
            ("maas_auth_policy_tinyllama_free", "MaaSAuthPolicy"),
            ("maas_subscription_tinyllama_free", "MaaSSubscription"),
        ],
    )
    def test_maas_subscription_stack_ready_for_free_model(
        self,
        request: pytest.FixtureRequest,
        resource_fixture: str,
        kind_label: str,
        maas_auth_policy_tinyllama_free: Any,
        maas_subscription_tinyllama_free: Any,
    ) -> None:
        """Verify the MaaS subscription flow objects are created and Ready."""
        resource = request.getfixturevalue(argname=resource_fixture)
        LOGGER.info(f"Checking {kind_label} {resource.name} is Ready")
        resource.wait_for_condition(condition="Ready", status="True", timeout=300)
