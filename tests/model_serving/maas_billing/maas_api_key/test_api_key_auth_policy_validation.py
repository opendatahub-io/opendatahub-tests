from __future__ import annotations

import pytest
import requests
import structlog
from pytest_testconfig import config as py_config

from tests.model_serving.maas_billing.maas_api_key.utils import get_auth_policy_callback_url
from tests.model_serving.maas_billing.utils import get_maas_models_response

LOGGER = structlog.get_logger(name=__name__)

MAAS_API_AUTH_POLICY_NAME = "maas-api-auth-policy"


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "minimal_subscription_for_free_user",
)
class TestAuthPolicyApiKeyValidation:
    """Verify the maas-api-auth-policy callback URL uses the correct namespace."""

    @pytest.mark.smoke
    def test_auth_policy_callback_url_uses_correct_namespace(
        self,
        admin_client,
    ) -> None:
        """Verify the apiKeyValidation callback URL does not reference the wrong namespace."""
        callback_url = get_auth_policy_callback_url(
            admin_client=admin_client,
            policy_name=MAAS_API_AUTH_POLICY_NAME,
            namespace=py_config["applications_namespace"],
        )

        expected_host = f"maas-api.{py_config['applications_namespace']}.svc.cluster.local"
        assert expected_host in callback_url, (
            f"apiKeyValidation callback URL uses wrong namespace. "
            f"Expected '{expected_host}' in URL, got: {callback_url}"
        )

        LOGGER.info(
            f"AuthPolicy callback URL correctly uses namespace '{py_config['applications_namespace']}': {callback_url}"
        )

    @pytest.mark.smoke
    @pytest.mark.usefixtures(
        "maas_model_tinyllama_free",
        "maas_auth_policy_tinyllama_free",
    )
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_api_key_can_list_models(
        self,
        request_session_http: requests.Session,
        base_url: str,
        api_key_for_free_model_listing: str,
    ) -> None:
        """Verify an API key can list models via GET /v1/models without 403."""
        models_response = get_maas_models_response(
            session=request_session_http,
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key_for_free_model_listing}",
                "Content-Type": "application/json",
            },
        )
        models_list: list[dict] = models_response.json().get("data", [])
        assert models_list, "Expected at least one model from /v1/models"

        LOGGER.info(f"API key successfully listed {len(models_list)} model(s) via /v1/models")
