import pytest
import requests
from typing import Self, Any
from tests.model_registry.model_registry.rest_api.utils import register_model_rest_api, validate_resource_attributes
from tests.model_registry.utils import get_mr_service_by_label, get_endpoint_from_mr_service
from kubernetes.dynamic import DynamicClient
from utilities.constants import Protocols
from utilities.resources.model_registry_modelregistry_opendatahub_io import ModelRegistry

from simple_logger.logger import get_logger


LOGGER = get_logger(name=__name__)


@pytest.mark.parametrize(
    "db_backend_under_test,model_registry_metadata_db_resources",
    [
        pytest.param("mysql", {"db_name": "mysql"}),
        pytest.param("postgres", {"db_name": "postgres"}),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("db_backend_under_test", "model_registry_metadata_db_resources")
class TestModelRegistryWithSecureDB:
    """
    Test suite for validating Model Registry functionality with secure database connections (SSL/TLS).
    Tests both MySQL and PostgreSQL backends with invalid and valid CA certificate scenarios.
    """

    # Implements RHOAIENG-26150
    @pytest.mark.parametrize(
        "patch_external_deployment_with_ssl_ca,patch_invalid_ca,local_ca_bundle",
        [
            (
                {
                    "ca_configmap_name": "odh-trusted-ca-bundle",
                    "ca_mount_path": "/etc/mysql/ssl",
                    "ca_configmap_for_test": False,
                },
                {"ca_configmap_name": "odh-trusted-ca-bundle", "ca_file_name": "invalid-ca.crt"},
                {"cert_name": "invalid-ca.crt"},
            ),
        ],
        indirect=True,
    )
    @pytest.mark.usefixtures(
        "deploy_secure_db_mr",
        "patch_external_deployment_with_ssl_ca",
        "patch_invalid_ca",
    )
    def test_register_model_with_invalid_ca(
        self: Self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        model_registry_rest_headers: dict[str, str],
        local_ca_bundle: str,
        deploy_secure_db_mr: ModelRegistry,
        model_data_for_test: dict[str, Any],
    ):
        """
        Test that model registration fails with an SSLError when the Model Registry is deployed
        with an invalid CA certificate.
        """
        service = get_mr_service_by_label(
            client=admin_client, namespace_name=model_registry_namespace, mr_instance=deploy_secure_db_mr
        )
        model_registry_rest_url = get_endpoint_from_mr_service(svc=service, protocol=Protocols.REST)

        with pytest.raises(requests.exceptions.SSLError) as exc_info:
            register_model_rest_api(
                model_registry_rest_url=f"https://{model_registry_rest_url}",
                model_registry_rest_headers=model_registry_rest_headers,
                data_dict=model_data_for_test,
                verify=local_ca_bundle,
            )
        assert "SSLError" in str(exc_info.value), (
            f"Expected SSL certificate verification failure, got: {exc_info.value}"
        )

    @pytest.mark.parametrize(
        "patch_external_deployment_with_ssl_ca, deploy_secure_db_mr,local_ca_bundle",
        [
            (
                {
                    "ca_configmap_name": "db-ca-configmap",
                    "ca_mount_path": "/etc/mysql/ssl",
                    "ca_configmap_for_test": True,
                },
                {"sslRootCertificateConfigMap": {"name": "db-ca-configmap", "key": "ca-bundle.crt"}},
                {"cert_name": "ca-bundle.crt"},
            ),
        ],
        indirect=True,
    )
    @pytest.mark.usefixtures(
        "model_registry_metadata_db_resources",
        "deploy_secure_db_mr",
        "ca_configmap_for_test",
        "patch_external_deployment_with_ssl_ca",
    )
    @pytest.mark.smoke
    def test_register_model_with_valid_ca(
        self: Self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        model_registry_rest_headers: dict[str, str],
        local_ca_bundle: str,
        deploy_secure_db_mr: ModelRegistry,
        model_data_for_test: dict[str, Any],
        db_backend_under_test: str,
    ):
        service = get_mr_service_by_label(
            client=admin_client, namespace_name=model_registry_namespace, mr_instance=deploy_secure_db_mr
        )
        model_registry_rest_url = get_endpoint_from_mr_service(svc=service, protocol=Protocols.REST)

        result = register_model_rest_api(
            model_registry_rest_url=f"https://{model_registry_rest_url}",
            model_registry_rest_headers=model_registry_rest_headers,
            data_dict=model_data_for_test,
            verify=local_ca_bundle,
        )
        assert result["register_model"].get("id"), f"Model registration failed with secure DB {db_backend_under_test}."
        validate_resource_attributes(
            expected_params=model_data_for_test["register_model_data"],
            actual_resource_data=result["register_model"],
            resource_name="register_model",
        )
        LOGGER.info(f"Model registered successfully with secure DB {db_backend_under_test} using {local_ca_bundle}")
