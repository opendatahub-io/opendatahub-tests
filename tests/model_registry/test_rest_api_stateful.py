import pytest
from simple_logger.logger import get_logger
from tests.model_registry.constants import MR_NAMESPACE
from utilities.constants import DscComponents

LOGGER = get_logger(name=__name__)


@pytest.mark.fuzzer
@pytest.mark.sanity
@pytest.mark.parametrize(
    "model_registry_namespace, updated_dsc_component_state_scope_class",
    [
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
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("model_registry_namespace", "updated_dsc_component_state_scope_class")
class TestRestAPIStateful:
    def test_mr_api(self, state_machine):
        state_machine.run()
