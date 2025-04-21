import pytest
from simple_logger.logger import get_logger
from tests.model_registry.constants import MR_NAMESPACE
from utilities.constants import DscComponents

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
                        "registriesNamespace": MR_NAMESPACE,
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
