# AI Disclaimer: Google Gemini 2.5 pro has been used to generate a majority of this code, with human review and editing.
from typing import Any, Self

import pytest
from model_registry import ModelRegistry as ModelRegistryClient
from mr_openapi.exceptions import ForbiddenException, UnauthorizedException
from ocp_resources.service_account import ServiceAccount
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler, retry

from tests.model_registry.model_registry.rbac.utils import build_mr_client_args
from utilities.infra import create_inference_token

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

        # Retry for up to 2 minutes if we get UnauthorizedException (401) during kube-rbac-proxy initialization
        # Expect ForbiddenException (403) once kube-rbac-proxy is fully initialized
        http_error = _try_connection_expect_forbidden(client_args=client_args)

        # Verify the status code from the caught exception
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

        # Create a fresh token to bypass kube-rbac-proxy cache from previous test
        fresh_token = create_inference_token(model_service_account=service_account)
        client_args = build_mr_client_args(
            rest_endpoint=model_registry_instance_rest_endpoint[0], token=fresh_token, author="rbac-test-granted"
        )
        LOGGER.debug(f"Attempting client connection with args: {client_args}")

        # Retry for up to 2 minutes to allow RBAC propagation
        # Accept UnauthorizedException (401) as a transient error during RBAC propagation
        sampler = TimeoutSampler(
            wait_timeout=120,
            sleep=5,
            func=lambda: ModelRegistryClient(**client_args),
            exceptions_dict={UnauthorizedException: []},
        )

        try:
            mr_client_success = None
            # Get the first successful result
            for mr_client_success in sampler:
                if mr_client_success:
                    break
            assert mr_client_success is not None, "Client initialization failed after granting permissions"
            LOGGER.info("Client instantiated successfully after granting permissions.")
        except Exception as e:
            LOGGER.error(f"Failed to access Model Registry after granting permissions: {e}", exc_info=True)
            raise

        LOGGER.info("--- RBAC Test Completed Successfully ---")


@retry(wait_timeout=120, sleep=5, exceptions_dict={UnauthorizedException: []})
def _try_connection_expect_forbidden(client_args: dict[str, Any]) -> ForbiddenException:
    """
    Attempts to create a ModelRegistryClient and expects ForbiddenException.
    Retries on UnauthorizedException (401) during kube-rbac-proxy initialization.
    Returns the ForbiddenException when received.
    """
    try:
        ModelRegistryClient(**client_args)
        raise AssertionError("Expected ForbiddenException but client connection succeeded")
    except ForbiddenException as e:
        # This is what we want - 403 Forbidden
        return e
