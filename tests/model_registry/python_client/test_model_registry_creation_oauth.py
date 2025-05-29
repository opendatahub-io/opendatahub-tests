import pytest
from typing import Self
from simple_logger.logger import get_logger
from pytest_testconfig import config as py_config

from ocp_resources.namespace import Namespace
from utilities.constants import DscComponents, Protocols
from tests.model_registry.constants import MODEL_NAME, MODEL_DICT
from model_registry import ModelRegistry as ModelRegistryClient
from tests.model_registry.utils import get_endpoint_from_mr_service, get_mr_service_by_label

LOGGER = get_logger(name=__name__)


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
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("updated_dsc_component_state_scope_class")
class TestModelRegistryCreationOAuth:
    """
    Tests the creation of a model registry with OAuth proxy configuration.
    """

    @pytest.mark.smoke
    def test_registering_model_with_oauth(
        self: Self,
        admin_client,
        model_registry_namespace: str,
        model_registry_instance_oauth_proxy,
        current_client_token: str,
    ):
        # Get the service for the OAuth proxy configured model registry
        mr_service = get_mr_service_by_label(
            client=admin_client,
            ns=Namespace(name=model_registry_namespace),
            mr_instance=model_registry_instance_oauth_proxy,
        )

        # Get the REST endpoint
        rest_endpoint = get_endpoint_from_mr_service(svc=mr_service, protocol=Protocols.REST)

        # Create client with OAuth proxy endpoint
        server, port = rest_endpoint.split(":")
        client = ModelRegistryClient(
            server_address=f"{Protocols.HTTPS}://{server}",
            port=port,
            author="opendatahub-test",
            user_token=current_client_token,
            is_secure=False,
        )

        # Register a new model
        registered_model = client.register_model(
            name=MODEL_DICT["model_name"],
            uri=MODEL_DICT["model_uri"],
            version=MODEL_DICT["model_version"],
            description=MODEL_DICT["model_description"],
            model_format_name=MODEL_DICT["model_format"],
            model_format_version=MODEL_DICT["model_format_version"],
            storage_key=MODEL_DICT["model_storage_key"],
            storage_path=MODEL_DICT["model_storage_path"],
            metadata=MODEL_DICT["model_metadata"],
        )

        # Get and verify the model
        model = client.get_registered_model(name=MODEL_NAME)
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
