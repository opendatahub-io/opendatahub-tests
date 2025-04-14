import pytest
from typing import Self
from simple_logger.logger import get_logger

from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.pod import Pod
from ocp_resources.namespace import Namespace
from utilities.constants import DscComponents
from tests.model_registry.constants import MR_NAMESPACE, MODEL_NAME, MODEL_DICT
from model_registry import ModelRegistry as ModelRegistryClient
from model_registry.types import RegisteredModel
from kubernetes.dynamic import DynamicClient

LOGGER = get_logger(name=__name__)

CUSTOM_NAMESPACE = "model-registry-custom-ns"


@pytest.mark.parametrize(
    "model_registry_namespace, updated_dsc_component_state_scope_class",
    [
        pytest.param(
            {"namespace_name": CUSTOM_NAMESPACE},
            {
                "component_patch": {
                    DscComponents.MODELREGISTRY: {
                        "managementState": DscComponents.ManagementState.MANAGED,
                        "registriesNamespace": CUSTOM_NAMESPACE,
                    },
                }
            },
        ),
        pytest.param(
            {"namespace_name": MR_NAMESPACE},
            {
                "component_patch": {
                    DscComponents.MODELREGISTRY: {
                        "managementState": DscComponents.ManagementState.MANAGED,
                        "registriesNamespace": MR_NAMESPACE,
                    },
                },
            },
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("model_registry_namespace", "updated_dsc_component_state_scope_class")
class TestModelRegistryCreation:
    """
    Tests the creation of a model registry. If the component is set to 'Removed' it will be switched to 'Managed'
    for the duration of this test module.
    """

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "registered_model",
        [
            pytest.param(
                MODEL_DICT,
            )
        ],
        indirect=True,
    )
    def test_registering_model(
        self: Self,
        model_registry_client: ModelRegistryClient,
        registered_model: RegisteredModel,
    ):
        model = model_registry_client.get_registered_model(MODEL_NAME)
        expected_attrs = {
            "id": registered_model.id,
            "name": registered_model.name,
            "description": registered_model.description,
            "owner": registered_model.owner,
            "state": registered_model.state,
        }
        errors = [
            f"Unexpected {attr} expected: {expected}, received {getattr(model, attr)}"
            for attr, expected in expected_attrs.items()
            if getattr(model, attr) != expected
        ]
        if errors:
            pytest.fail("errors found in model registry response validation:\n{}".format("\n".join(errors)))

    def test_model_registry_operator_env(
        self,
        admin_client: DynamicClient,
        model_registry_namespace: Namespace,
        updated_dsc_component_state_scope_class: DataScienceCluster,
        model_registry_operator_pod: Pod,
    ):
        namespace_env = []
        for container in model_registry_operator_pod.instance.spec.containers:
            for env in container.env:
                if env.name == "REGISTRIES_NAMESPACE" and env.value == model_registry_namespace.name:
                    namespace_env.append({container.name: env})
        if not namespace_env:
            pytest.fail("Missing environment variable REGISTRIES_NAMESPACE")

    # TODO: Edit a registered model
    # TODO: Add additional versions for a model
    # TODO: List all available models
    # TODO: List all versions of a model
