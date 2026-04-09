from __future__ import annotations

import pytest
import structlog
from pytest_testconfig import config as py_config

from tests.model_serving.maas_billing.maas_api_key.utils import get_auth_policy_callback_url

LOGGER = structlog.get_logger(name=__name__)

MAAS_API_AUTH_POLICY_NAME = "maas-api-auth-policy"


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
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
