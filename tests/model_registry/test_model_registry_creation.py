import shlex
from typing import Self
from simple_logger.logger import get_logger
from ocp_resources.model_registry import ModelRegistry
from pyhelper_utils.shell import run_command

from tests.model_registry.utils import generate_register_model_command

LOGGER = get_logger(name=__name__)


class TestModelRegistryCreation:
    """Tests the creation of a model registry"""

    # TODO: Enable Model Registry in DSC if needed

    def test_model_registry_instance_creation(self: Self, model_registry_instance: ModelRegistry):
        assert model_registry_instance.name == "model-registry"

    def test_registering_model(self: Self, model_registry_instance_rest_endpoint: str, admin_client_token: str):
        cmd = generate_register_model_command(model_registry_instance_rest_endpoint, admin_client_token)
        res, out, _ = run_command(command=shlex.split(cmd), verify_stderr=False, check=False)
        assert res
        out_dict = eval(out)
        assert out_dict["name"] == "model-name"
        assert out_dict["description"] == "test-model"
        assert out_dict["externalId"] == "1"
        assert out_dict["owner"] == "opendatahub-tests-client"
        assert out_dict["state"] == "LIVE"

    # TODO: Query for a registered Model
