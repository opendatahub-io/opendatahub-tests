import pytest
from typing import Self
from tests.model_registry.rest_api.utils import register_model_rest_api
from tests.model_registry.rest_api.constants import MODEL_REGISTER_DATA
from utilities.constants import DscComponents
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)


@pytest.mark.parametrize(
    "updated_dsc_component_state_scope_class",
    [
        pytest.param(
            {
                "component_patch": {
                    DscComponents.MODELREGISTRY: {
                        "managementState": DscComponents.ManagementState.MANAGED,
                        "registriesNamespace": "test-model-registry-namespace",
                    },
                },
            },
        ),
    ],
    indirect=True,
)
@pytest.mark.jira("RHOAIENG-26150")
@pytest.mark.usefixtures(
    "updated_dsc_component_state_scope_class",
    "deploy_secure_mysql_and_mr",
)
class TestModelRegistryWithSecureDB:
    """
    Test suite for validating Model Registry functionality with a secure MySQL database connection (SSL/TLS).
    """

    def test_register_model_with_secure_db(
        self: Self,
        model_registry_rest_url: str,
        model_registry_rest_headers: dict[str, str],
    ):
        """
        Deploys Model Registry with a secure MySQL DB (SSL/TLS), registers a model, and checks functionality.
        Assumes the fixture 'deploy_secure_mysql_and_mr' sets up MySQL and MR with SSL certs and config.
        """
        result = register_model_rest_api(
            model_registry_rest_url=model_registry_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            data_dict=MODEL_REGISTER_DATA,
        )
        assert result["register_model"].get("id"), "Model registration failed with secure DB connection."
        LOGGER.info(f"Model registered successfully with secure DB: {result['register_model']['id']}")
