# AI Disclaimer: Google Gemini 2.5 pro has been used to generate a majority of this code, with human review and editing.
import pytest
from typing import Self
from simple_logger.logger import get_logger
from model_registry import ModelRegistry as ModelRegistryClient
from tests.model_registry.rbac.utils import build_mr_client_args
from utilities.infra import create_inference_token
from mr_openapi.exceptions import ForbiddenException
from ocp_resources.service_account import ServiceAccount

LOGGER = get_logger(name=__name__)


@pytest.mark.usefixtures(
    "updated_dsc_component_state_scope_session",
    "model_registry_namespace",
    "model_registry_metadata_db_resources",
    "model_registry_instance",
)
@pytest.mark.custom_namespace
class TestModelRegistryRBAC:
    """
    Tests RBAC for Model Registry REST endpoint using ServiceAccount tokens.
    Tests both standard and OAuth proxy configurations.
    """

    @pytest.mark.sanity
    @pytest.mark.usefixtures("sa_namespace", "service_account")
    def test_service_account_access_denied(
        self: Self,
        model_registry_instance_rest_endpoint: list[str],
        sa_token: str,
    ):
        """
        Verifies SA access is DENIED (403 Forbidden) by default via REST.
        Does NOT use mr_access_role or mr_access_role_binding fixtures.
        """
        LOGGER.info("--- Starting RBAC Test: Access Denied ---")
        LOGGER.info(f"Targeting Model Registry REST endpoint: {model_registry_instance_rest_endpoint}")
        LOGGER.info("Expecting initial access DENIAL (403 Forbidden)")

        client_args = build_mr_client_args(
            rest_endpoint=model_registry_instance_rest_endpoint[0], token=sa_token, author="rbac-test-denied"
        )
        LOGGER.debug(f"Attempting client connection with args: {client_args}")

        # Expect an exception related to HTTP 403
        with pytest.raises(ForbiddenException) as exc_info:
            _ = ModelRegistryClient(**client_args)

        # Verify the status code from the caught exception
        http_error = exc_info.value
        assert http_error.body is not None, "HTTPError should have a response object"
        LOGGER.info(f"Received expected HTTP error: Status Code {http_error.status}")
        assert http_error.status == 403, f"Expected HTTP 403 Forbidden, but got {http_error.status}"
        LOGGER.info("Successfully received expected HTTP 403 status code.")

    @pytest.mark.smoke
    @pytest.mark.usefixtures("sa_namespace", "service_account", "mr_access_role", "mr_access_role_binding")
    def test_service_account_access_granted(
        self: Self,
        service_account: ServiceAccount,
        model_registry_instance_rest_endpoint: list[str],
    ):
        """
        Verifies SA access is GRANTED via REST after applying Role and RoleBinding fixtures.
        """
        LOGGER.info("--- Starting RBAC Test: Access Granted ---")
        LOGGER.info(f"Targeting Model Registry REST endpoint: {model_registry_instance_rest_endpoint[0]}")
        LOGGER.info("Applied RBAC Role/Binding via fixtures. Expecting access GRANT.")

        # Create a fresh token to bypass OAuth proxy cache from previous test
        fresh_token = create_inference_token(model_service_account=service_account)

        try:
            client_args = build_mr_client_args(
                rest_endpoint=model_registry_instance_rest_endpoint[0], token=fresh_token, author="rbac-test-granted"
            )
            LOGGER.debug(f"Attempting client connection with args: {client_args}")
            mr_client_success = ModelRegistryClient(**client_args)
            assert mr_client_success is not None, "Client initialization failed after granting permissions"
            LOGGER.info("Client instantiated successfully after granting permissions.")

        except Exception as e:
            # If we get an exception here, it's unexpected, especially 403
            LOGGER.error(f"Received unexpected general error after granting access: {e}", exc_info=True)
            raise

        LOGGER.info("--- RBAC Test Completed Successfully ---")
