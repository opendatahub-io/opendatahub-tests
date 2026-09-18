import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.deployment import Deployment

from tests.ai_safety.trustyai_module.constants import TRUSTYAI_MODULE_CRD_NAME
from tests.ai_safety.trustyai_module.utils import (
    validate_configmap_exists,
    validate_deployment_available,
    validate_trustyai_module_observed_generation,
    validate_trustyai_module_provisioning_succeeded,
    validate_trustyai_module_ready,
)
from utilities.resources.trustyai_module import TrustyAI


@pytest.mark.tier1
@pytest.mark.ai_safety
@pytest.mark.component_health
@pytest.mark.usefixtures("modular_trustyai_architecture")
class TestTrustyAIModuleHealth:
    """Tier 1 health checks for the TrustyAI modular operator stack."""

    def test_trustyai_module_crd_exists(self, admin_client: DynamicClient) -> None:
        """Given TrustyAI is enabled on the DSC, the module CRD is installed."""
        crd = CustomResourceDefinition(
            client=admin_client,
            name=TRUSTYAI_MODULE_CRD_NAME,
            ensure_exists=True,
        )
        assert crd.exists, f"CRD {TRUSTYAI_MODULE_CRD_NAME} does not exist on the cluster"

    def test_trustyai_module_operator_deployment_available(
        self,
        trustyai_module_operator_deployment: Deployment,
    ) -> None:
        """Given modular TrustyAI is enabled, the module operator Deployment is Available."""
        validate_deployment_available(deployment=trustyai_module_operator_deployment)

    def test_trustyai_workload_operator_deployment_available(
        self,
        trustyai_workload_operator_deployment: Deployment,
    ) -> None:
        """Given the module operator reconciled, the workload operator Deployment is Available."""
        validate_deployment_available(deployment=trustyai_workload_operator_deployment)

    def test_trustyai_module_cr_ready(self, trustyai_module: TrustyAI) -> None:
        """Given platform and module operators reconciled, the singleton module CR is Ready."""
        validate_trustyai_module_ready(trustyai_module=trustyai_module)
        validate_trustyai_module_provisioning_succeeded(trustyai_module=trustyai_module)
        validate_trustyai_module_observed_generation(trustyai_module=trustyai_module)

    def test_trustyai_platform_and_dsc_configmaps_exist(
        self,
        trustyai_platform_configmap: ConfigMap,
        trustyai_dsc_configmap: ConfigMap,
    ) -> None:
        """Given the module reconciled, platform and DSC ConfigMaps are present."""
        validate_configmap_exists(configmap=trustyai_platform_configmap)
        validate_configmap_exists(configmap=trustyai_dsc_configmap)
