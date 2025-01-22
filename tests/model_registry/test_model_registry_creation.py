import pytest
from typing import Self
from simple_logger.logger import get_logger

from utilities.constants import Protocols
from model_registry import ModelRegistry

LOGGER = get_logger(name=__name__)


class TestModelRegistryCreation:
    """Tests the creation of a model registry"""

    # TODO: Enable Model Registry in DSC if needed

    @pytest.mark.smoke
    def test_registering_model(self: Self, model_registry_instance_rest_endpoint: str, current_client_token: str):
        # address and port need to be split in the client instantiation
        server = model_registry_instance_rest_endpoint.split(":")[0]
        port = model_registry_instance_rest_endpoint.split(":")[1]
        registry = ModelRegistry(
            server_address=f"{Protocols.HTTPS}://{server}",
            port=port,
            author="opendatahub-test",
            user_token=current_client_token,
            is_secure=False,
        )
        model = registry.register_model(
            name="my-model",  # model name
            uri="https://storage-place.my-company.com",  # model URI
            version="2.0.0",
            description="lorem ipsum",
            model_format_name="onnx",
            model_format_version="1",
            storage_key="my-data-connection",
            storage_path="path/to/model",
            metadata={
                # can be one of the following types
                "int_key": 1,
                "bool_key": False,
                "float_key": 3.14,
                "str_key": "str_value",
            },
        )
        registered_model = registry.get_registered_model("my-model")
        errors = []
        if not registered_model.id == model.id:
            errors.append(f"Unexpected id, received {registered_model.id}")
        if not registered_model.name == model.name:
            errors.append(f"Unexpected name, received {registered_model.name}")
        if not registered_model.description == model.description:
            errors.append(f"Unexpected description, received {registered_model.description}")
        if not registered_model.owner == model.owner:
            errors.append(f"Unexpected owner, received {registered_model.owner}")
        if not registered_model.state == model.state:
            errors.append(f"Unexpected state, received {registered_model.state}")

        assert not errors, "errors occured:\n{}".format("\n".join(errors))

    # TODO: Edit a registered model
    # TODO: Add additional versions for a model
    # TODO: List all available models
    # TODO: List all versions of a model
