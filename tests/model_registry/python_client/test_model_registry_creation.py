import pytest
from typing import Self
from simple_logger.logger import get_logger

from ocp_resources.pod import Pod
from ocp_resources.namespace import Namespace
from utilities.constants import DscComponents
from tests.model_registry.constants import MODEL_NAME, MODEL_DICT, MR_NAMESPACE
from model_registry import ModelRegistry as ModelRegistryClient
from model_registry.types import RegisteredModel

LOGGER = get_logger(name=__name__)

CUSTOM_NAMESPACE = "model-registry-custom-ns"


@pytest.mark.parametrize(
    "updated_dsc_component_state_scope_class, registered_model",
    [
        pytest.param(
            {
                "component_patch": {
                    DscComponents.MODELREGISTRY: {
                        "managementState": DscComponents.ManagementState.MANAGED,
                        "registriesNamespace": CUSTOM_NAMESPACE,
                    },
                }
            },
            MODEL_DICT,
        ),
        pytest.param(
            {
                "component_patch": {
                    DscComponents.MODELREGISTRY: {
                        "managementState": DscComponents.ManagementState.MANAGED,
                        "registriesNamespace": MR_NAMESPACE,
                    },
                },
            },
            MODEL_DICT,
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("updated_dsc_component_state_scope_class", "registered_model")
class TestModelRegistryCreation:
    """
    Tests the creation of a model registry. If the component is set to 'Removed' it will be switched to 'Managed'
    for the duration of this test module.
    """

    @pytest.mark.smoke
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
        updated_dsc_component_state_scope_class: Namespace,
        model_registry_namespace: str,
        model_registry_operator_pod: Pod,
    ):
        namespace_env = []
        for container in model_registry_operator_pod.instance.spec.containers:
            for env in container.env:
                if env.name == "REGISTRIES_NAMESPACE" and env.value == model_registry_namespace:
                    namespace_env.append({container.name: env})
        if not namespace_env:
            pytest.fail("Missing environment variable REGISTRIES_NAMESPACE")

    # TODO: Edit a registered model
    # TODO: Add additional versions for a model
    # TODO: List all available models
    # TODO: List all versions of a model
