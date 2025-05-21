import pytest
from simple_logger.logger import get_logger
from utilities.constants import DscComponents
from pytest_testconfig import config as py_config

LOGGER = get_logger(name=__name__)


@pytest.mark.fuzzer
@pytest.mark.parametrize(
    "updated_dsc_component_state_scope_class",
    [
        pytest.param(
            {
                "component_patch": {
                    DscComponents.MODELREGISTRY: {
                        "managementState": DscComponents.ManagementState.MANAGED,
                        "registriesNamespace": py_config["model_registry_namespace"],
                    },
                },
            },
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("updated_dsc_component_state_scope_class")
class TestRestAPIStateful:
    def test_mr_api_stateful(self, state_machine):
        """Launches stateful tests against the Model Registry API endpoints defined in its openAPI yaml spec file"""
        state_machine.run()
