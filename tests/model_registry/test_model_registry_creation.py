import pytest
from typing import Self
from model_registry import ModelRegistry as ModelRegistryClient

from simple_logger.logger import get_logger

from tests.model_registry.constants import MODEL_NAME, MR_NAMESPACE
from tests.model_registry.utils import register_model, get_and_validate_registered_model
from utilities.constants import DscComponents


LOGGER = get_logger(name=__name__)


@pytest.mark.parametrize(
    "updated_dsc_component_state_scope_class",
    [
        pytest.param(
            {
                "component_patch": {
                    DscComponents.MODELREGISTRY: {
                        "managementState": DscComponents.ManagementState.MANAGED,
                        "registriesNamespace": MR_NAMESPACE,
                    },
                }
            },
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("updated_dsc_component_state_scope_class")
class TestModelRegistryCreation:
    """
    Tests the creation of a model registry. If the component is set to 'Removed' it will be switched to 'Managed'
    for the duration of this test module.
    """

    @pytest.mark.smoke
    def test_registering_model(
        self: Self,
        model_registry_client: ModelRegistryClient,
    ):
        model = register_model(model_registry_client=model_registry_client)
        get_and_validate_registered_model(
            model_registry_client=model_registry_client, model_name=MODEL_NAME, registered_model=model
        )

    @pytest.mark.smoke
    def test_registering_modelfail(self):
        pytest.fail("test failure")