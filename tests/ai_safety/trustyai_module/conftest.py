import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.deployment import Deployment
from pytest_testconfig import config as py_config

from tests.ai_safety.trustyai_module.constants import (
    TRUSTYAI_DSC_CONFIGMAP,
    TRUSTYAI_MODULE_CRD_NAME,
    TRUSTYAI_MODULE_NAME,
    TRUSTYAI_MODULE_OPERATOR_DEPLOYMENT,
    TRUSTYAI_PLATFORM_CONFIGMAP,
    TRUSTYAI_WORKLOAD_OPERATOR_DEPLOYMENT,
)
from utilities.resources.trustyai_module import TrustyAI


@pytest.fixture(scope="session")
def modular_trustyai_architecture(admin_client: DynamicClient) -> None:
    """Skip when the cluster has not migrated TrustyAI to the modular operator path."""
    module_crd = CustomResourceDefinition(client=admin_client, name=TRUSTYAI_MODULE_CRD_NAME)
    if not module_crd.exists:
        pytest.skip(
            f"TrustyAI module CRD {TRUSTYAI_MODULE_CRD_NAME} is not installed; "
            "modular architecture tests require the platform module handler."
        )

    module_operator = Deployment(
        client=admin_client,
        name=TRUSTYAI_MODULE_OPERATOR_DEPLOYMENT,
        namespace=py_config["applications_namespace"],
    )
    if not module_operator.exists:
        pytest.skip(
            f"Deployment {TRUSTYAI_MODULE_OPERATOR_DEPLOYMENT} is not present in "
            f"{py_config['applications_namespace']}; modular architecture is not enabled."
        )


@pytest.fixture(scope="session")
def trustyai_module_operator_deployment(
    admin_client: DynamicClient,
    modular_trustyai_architecture: None,
) -> Deployment:
    return Deployment(
        client=admin_client,
        name=TRUSTYAI_MODULE_OPERATOR_DEPLOYMENT,
        namespace=py_config["applications_namespace"],
        ensure_exists=True,
    )


@pytest.fixture(scope="session")
def trustyai_workload_operator_deployment(
    admin_client: DynamicClient,
    modular_trustyai_architecture: None,
) -> Deployment:
    return Deployment(
        client=admin_client,
        name=TRUSTYAI_WORKLOAD_OPERATOR_DEPLOYMENT,
        namespace=py_config["applications_namespace"],
        ensure_exists=True,
    )


@pytest.fixture(scope="session")
def trustyai_module(
    admin_client: DynamicClient,
    modular_trustyai_architecture: None,
) -> TrustyAI:
    return TrustyAI(
        client=admin_client,
        name=TRUSTYAI_MODULE_NAME,
        ensure_exists=True,
    )


@pytest.fixture(scope="session")
def trustyai_platform_configmap(
    admin_client: DynamicClient,
    modular_trustyai_architecture: None,
) -> ConfigMap:
    return ConfigMap(
        client=admin_client,
        name=TRUSTYAI_PLATFORM_CONFIGMAP,
        namespace=py_config["applications_namespace"],
        ensure_exists=True,
    )


@pytest.fixture(scope="session")
def trustyai_dsc_configmap(
    admin_client: DynamicClient,
    modular_trustyai_architecture: None,
) -> ConfigMap:
    return ConfigMap(
        client=admin_client,
        name=TRUSTYAI_DSC_CONFIGMAP,
        namespace=py_config["applications_namespace"],
        ensure_exists=True,
    )
