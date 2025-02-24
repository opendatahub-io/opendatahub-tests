import pytest
from typing import Self
from simple_logger.logger import get_logger

from ocp_resources.data_science_cluster import DataScienceCluster
from utilities.constants import Protocols, DscComponents, ModelFormat
from model_registry import ModelRegistry


LOGGER = get_logger(name=__name__)
MODEL_NAME: str = "my-model"


@pytest.mark.parametrize(
    "updated_dsc_component_state",
    [
        pytest.param(
            {
                "component_name": DscComponents.MODELREGISTRY,
                "desired_state": DscComponents.ManagementState.MANAGED,
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

    @pytest.mark.smoke
    def test_registering_model(
        self: Self,
        model_registry_instance_rest_endpoint: str,
        current_client_token: str,
        updated_dsc_component_state: DataScienceCluster,
    ):
        # address and port need to be split in the client instantiation
        server, port = model_registry_instance_rest_endpoint.split(":")
        registry = ModelRegistry(
            server_address=f"{Protocols.HTTPS}://{server}",
            port=port,
            author="opendatahub-test",
            user_token=current_client_token,
            is_secure=False,
        )
        model = registry.register_model(
            name=MODEL_NAME,
            uri="https://storage-place.my-company.com",
            version="2.0.0",
            description="lorem ipsum",
            model_format_name=ModelFormat.ONNX,
            model_format_version="1",
            storage_key="my-data-connection",
            storage_path="path/to/model",
            metadata={
                "int_key": 1,
                "bool_key": False,
                "float_key": 3.14,
                "str_key": "str_value",
            },
        )
        registered_model = registry.get_registered_model(MODEL_NAME)
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

        assert not errors, "errors found in model registry response validation:\n{}".format("\n".join(errors))

    # TODO: Edit a registered model
    # TODO: Add additional versions for a model
    # TODO: List all available models
    # TODO: List all versions of a model
