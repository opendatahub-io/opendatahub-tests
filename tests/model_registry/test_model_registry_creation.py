import pytest
import shlex
from ast import literal_eval
from typing import Self
from simple_logger.logger import get_logger
from pyhelper_utils.shell import run_command

from tests.model_registry.utils import generate_register_model_command
from utilities.constants import ComponentManagementState

LOGGER = get_logger(name=__name__)


@pytest.mark.parametrize(
    "updated_dsc_component_state",
    [
        pytest.param(
            {
                "component_name": "modelregistry",
                "desired_state": ComponentManagementState.MANAGED,
                "condition_type": "model-registry-operatorReady",
            },
        )
    ],
    indirect=True,
)
class TestModelRegistryCreation:
    """
    Tests the creation of a model registry. If the component is set to 'Removed' it will be switched to 'Managed'
    for the duration of this test module.
    """

    # TODO: Switch to Python client
    @pytest.mark.smoke
    def test_registering_model(
        self: Self,
        model_registry_instance_rest_endpoint: str,
        current_client_token: str,
        updated_dsc_component_state,
    ):
        cmd = generate_register_model_command(
            endpoint=model_registry_instance_rest_endpoint, token=current_client_token
        )
        _, out, _ = run_command(command=shlex.split(cmd))
        out_dict = literal_eval(node_or_string=out)
        errors = []
        if not out_dict["name"] == "model-name":
            errors.append(f"Unexpected name, received {out_dict['name']}")
        if not out_dict["description"] == "test-model":
            errors.append(f"Unexpected description, received {out_dict['description']}")
        if not out_dict["externalId"] == "1":
            errors.append(f"Unexpected id, received {out_dict['externalId']}")
        if not out_dict["owner"] == "opendatahub-tests-client":
            errors.append(f"Unexpected owner, received {out_dict['owner']}")
        if not out_dict["state"] == "LIVE":
            errors.append(f"Unexpected state, received {out_dict['state']}")

        assert not errors, "errors occured:\n{}".format("\n".join(errors))

    # TODO: Query for a registered Model
