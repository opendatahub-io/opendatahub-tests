import pytest
from typing import Self
from simple_logger.logger import get_logger
from pytest_testconfig import config as py_config
from semver import Version
from utilities.infra import get_product_version
from utilities.constants import DscComponents
from tests.model_registry.constants import MODEL_NAME, MODEL_DICT
from model_registry import ModelRegistry as ModelRegistryClient

LOGGER = get_logger(name=__name__)
MINVER = Version.parse(version="2.21.0")


@pytest.mark.parametrize(
    "updated_dsc_component_state_scope_class, model_registry_client",
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
            {"service_fixture": "model_registry_instance_oauth_service"},
            id="oauth_proxy",
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("updated_dsc_component_state_scope_class")
class TestModelRegistryCreationOAuth:
    """
    Tests the creation of a model registry with OAuth proxy configuration.
    Jira ID: RHOAIENG-26194
    """

    @pytest.mark.smoke
    def test_registering_model_with_oauth(
        self: Self,
        admin_client,
        model_registry_client: ModelRegistryClient,
    ):
        if py_config["distribution"] == "downstream" and get_product_version(admin_client=admin_client) < MINVER:
            pytest.skip("Skipping test for RHOAI < 2.21")

        # Register a new model
        registered_model = model_registry_client.register_model(
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
        model = model_registry_client.get_registered_model(name=MODEL_NAME)
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
