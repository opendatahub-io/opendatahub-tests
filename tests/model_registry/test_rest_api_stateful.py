import pytest
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)


@pytest.mark.fuzzer
@pytest.mark.sanity
@pytest.mark.parametrize(
    "updated_dsc_component_state",
    [
        pytest.param(
            {
                "component_name": "modelregistry",
                "desired_state": "Managed",
                "condition_type": "model-registry-operatorReady",
            },
        )
    ],
    indirect=True,
)
def test_mr_api(state_machine, updated_dsc_component_state):
    state_machine.run()
