import pytest
import requests
from typing import Self
from tests.model_registry.rest_api.utils import register_model_rest_api
from tests.model_registry.rest_api.constants import MODEL_REGISTER_DATA
from utilities.constants import DscComponents
from simple_logger.logger import get_logger
from tests.model_registry.utils import generate_random_name


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
@pytest.mark.usefixtures("updated_dsc_component_state_scope_class")
class TestModelRegistryWithSecureDB:
    """
    Test suite for validating Model Registry functionality with a secure MySQL database connection (SSL/TLS).
    Includes tests for both invalid and valid CA certificate scenarios.
    """

    # Implements RHOAIENG-26150
    @pytest.mark.usefixtures("model_registry_instance_ca")
    @pytest.mark.parametrize(
        "patch_invalid_ca",
        [{"ca_configmap_name": "odh-trusted-ca-bundle", "ca_file_name": "invalid-ca.crt"}],
        indirect=True,
    )
    def test_register_model_with_invalid_ca(
        self: Self,
        patch_invalid_ca: dict[str, str],
        model_registry_rest_url: str,
        model_registry_rest_headers: dict[str, str],
    ) -> None:
        """
        Test that model registration fails with an SSLError when the Model Registry is deployed
        with an invalid CA certificate.
        """
        model_name = generate_random_name(prefix="model-rest-api")
        MODEL_REGISTER_DATA["register_model_data"]["name"] = model_name
        with pytest.raises(requests.exceptions.SSLError) as exc_info:
            register_model_rest_api(
                model_registry_rest_url=model_registry_rest_url,
                model_registry_rest_headers=model_registry_rest_headers,
                data_dict=MODEL_REGISTER_DATA,
                verify=True,
            )
        assert "certificate verify failed" in str(exc_info.value), (
            f"Expected SSL certificate verification failure, got: {exc_info.value}"
        )

    # Implements RHOAIENG-26150
    @pytest.mark.usefixtures("deploy_secure_mysql_and_mr")
    @pytest.mark.parametrize(
        "local_ca_bundle",
        [{"cert_name": "ca-bundle.crt"}, {"cert_name": "odh-ca-bundle.crt"}],
        indirect=True,
    )
    def test_register_model_with_default_ca(
        self: Self,
        model_registry_rest_url: str,
        model_registry_rest_headers: dict[str, str],
        local_ca_bundle: str,
    ) -> None:
        """
        Deploys Model Registry with a secure MySQL DB (SSL/TLS), registers a model, and checks functionality.
        Uses a CA bundle file for SSL verification by passing it directly to the verify parameter.
        """
        model_name = generate_random_name(prefix="model-rest-api")
        MODEL_REGISTER_DATA["register_model_data"]["name"] = model_name
        result = register_model_rest_api(
            model_registry_rest_url=model_registry_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            data_dict=MODEL_REGISTER_DATA,
            verify=local_ca_bundle,
        )
        assert result["register_model"].get("id"), "Model registration failed with secure DB connection."
        for k, v in MODEL_REGISTER_DATA["register_model_data"].items():
            assert result["register_model"][k] == v, f"Expected {k}={v}, got {result[k]}"
        LOGGER.info(f"Model registered successfully with secure DB using {local_ca_bundle}")
