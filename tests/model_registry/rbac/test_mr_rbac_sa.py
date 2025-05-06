# AI Disclaimer: Google Gemini 2.5 pro has been used to generate a majority of this code, with human review and editing.
import pytest
from typing import Self, Dict, Any
from simple_logger.logger import get_logger

# Framework / Client specific imports
from model_registry import ModelRegistry as ModelRegistryClient
from mr_openapi.exceptions import ForbiddenException

from utilities.constants import DscComponents, Protocols  # Need Protocols.HTTPS
from tests.model_registry.constants import MR_NAMESPACE


LOGGER = get_logger(name=__name__)


# --- Helper to build client args ---
# Based on the 'model_registry_client' fixture
def build_mr_client_args(rest_endpoint: str, token: str, author: str) -> Dict[str, Any]:
    """Builds arguments for ModelRegistryClient based on REST endpoint and token."""
    try:
        server, port = rest_endpoint.split(":")

        # Conftest example uses is_secure=False - this implies TLS verification might be off
        # Adjust if your environment requires strict TLS verification
        return {
            "server_address": f"{Protocols.HTTPS}://{server}",
            "port": port,
            "user_token": token,
            "is_secure": False,  # Match conftest example (disable TLS verification)
            "author": author,
        }
    except Exception as e:
        LOGGER.error(f"Error parsing REST endpoint '{rest_endpoint}': {e}")
        raise ValueError(f"Could not parse REST endpoint: {rest_endpoint}") from e


# --- Test Class ---


# Parametrize to ensure Model Registry component is enabled via DSC fixture
@pytest.mark.parametrize(
    "updated_dsc_component_state_scope_class",
    [
        pytest.param(
            {
                "component_patch": {
                    DscComponents.MODELREGISTRY: {
                        "managementState": DscComponents.ManagementState.MANAGED,
                        "registriesNamespace": MR_NAMESPACE,
                    },
                }
            },
            id="enable_modelregistry_default_ns",
        )
    ],
    indirect=True,
    scope="class",
)
@pytest.mark.usefixtures("updated_dsc_component_state_scope_class")
class TestModelRegistryRBAC:
    """
    Tests RBAC for Model Registry REST endpoint using ServiceAccount tokens.
    """

    @pytest.mark.smoke
    @pytest.mark.usefixtures("sa_namespace", "service_account")  # Need SA and NS for token
    def test_service_account_access_denied(
        self: Self,
        sa_token: str,  # Get token for the test SA
        model_registry_instance_rest_endpoint: str,  # REST endpoint fixture
    ):
        """
        Verifies SA access is DENIED (403 Forbidden) by default via REST.
        Does NOT use mr_access_role or mr_access_role_binding fixtures.
        """
        LOGGER.info("--- Starting RBAC Test: Access Denied ---")
        LOGGER.info(f"Targeting Model Registry REST endpoint: {model_registry_instance_rest_endpoint}")
        LOGGER.info("Expecting initial access DENIAL (403 Forbidden)")

        try:
            client_args = build_mr_client_args(
                rest_endpoint=model_registry_instance_rest_endpoint, token=sa_token, author="rbac-test-denied"
            )
            LOGGER.debug(f"Attempting client connection with args: {client_args}")

            # Expect an exception related to HTTP 403
            # Adjust exception type based on ModelRegistryClient's behavior
            with pytest.raises(ForbiddenException) as exc_info:  # Or client's custom error e.g. ApiClientError
                _ = ModelRegistryClient(**client_args)

            # Verify the status code from the caught exception
            http_error = exc_info.value
            assert http_error.body is not None, "HTTPError should have a response object"
            LOGGER.info(f"Received expected HTTP error: Status Code {http_error.status}")
            assert http_error.status == 403, f"Expected HTTP 403 Forbidden, but got {http_error.status}"
            LOGGER.info("Successfully received expected HTTP 403 status code.")

        except Exception as e:
            LOGGER.error(f"Received unexpected error during 'access denied' check: {e}", exc_info=True)
            pytest.fail(f"'Access denied' check failed unexpectedly: {e}")

    @pytest.mark.smoke
    # Use fixtures for SA/NS/Token AND the RBAC Role/Binding
    @pytest.mark.usefixtures("sa_namespace", "service_account", "mr_access_role", "mr_access_role_binding")
    def test_service_account_access_granted(
        self: Self,
        sa_token: str,  # Get token for the test SA
        model_registry_instance_rest_endpoint: str,  # REST endpoint fixture
        # mr_access_role and mr_access_role_binding are activated by @usefixtures
    ):
        """
        Verifies SA access is GRANTED via REST after applying Role and RoleBinding fixtures.
        """
        LOGGER.info("--- Starting RBAC Test: Access Granted ---")
        LOGGER.info(f"Targeting Model Registry REST endpoint: {model_registry_instance_rest_endpoint}")
        LOGGER.info("Applied RBAC Role/Binding via fixtures. Expecting access GRANT.")

        try:
            client_args = build_mr_client_args(
                rest_endpoint=model_registry_instance_rest_endpoint, token=sa_token, author="rbac-test-granted"
            )
            LOGGER.debug(f"Attempting client connection with args: {client_args}")

            # Instantiate the client - this should now succeed or fail for non-auth reasons
            mr_client_success = ModelRegistryClient(**client_args)
            assert mr_client_success is not None, "Client initialization failed after granting permissions"
            LOGGER.info("Client instantiated successfully after granting permissions.")

        except ForbiddenException as e:
            # If we get an HTTP error here, it's unexpected, especially 403
            LOGGER.error(
                f"Received unexpected HTTP error after granting access: {e.status if e.body else 'No Response'} - {e}",
                exc_info=True,
            )
            if e.body is not None and e.status == 403:
                pytest.fail(f"Still received HTTP 403 Forbidden after applying Role/RoleBinding: {e}")
            else:
                pytest.fail(f"Client interaction failed with HTTP error after granting permissions: {e}")

        except Exception as e:
            LOGGER.error(f"Received unexpected general error after granting access: {e}", exc_info=True)
            pytest.fail(f"Client interaction failed unexpectedly after granting permissions: {e}")

        LOGGER.info("--- RBAC Test Completed Successfully ---")
