from __future__ import annotations

import pytest
import requests
import structlog
from ocp_resources.maas_subscription import MaaSSubscription

from tests.model_serving.maas_billing.maas_subscription.utils import fetch_and_assert_models_for_subscription
from tests.model_serving.maas_billing.utils import build_maas_headers

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "maas_model_tinyllama_free",
    "maas_auth_policy_tinyllama_free",
    "maas_subscription_tinyllama_free",
    "maas_model_tinyllama_premium",
    "maas_auth_policy_tinyllama_premium",
    "maas_subscription_tinyllama_premium",
)
class TestModelsEndpointAccessControl:
    """Security, RBAC, auto-selection, and rate-limit exemption tests for GET /v1/models."""

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_api_key_ignores_subscription_header(
        self,
        request_session_http: requests.Session,
        models_url: str,
        api_key_bound_to_free_subscription: str,
        maas_subscription_tinyllama_free: MaaSSubscription,
        maas_subscription_tinyllama_premium: MaaSSubscription,
    ) -> None:
        """Verify API key ignores client-injected X-MaaS-Subscription header."""
        expected_sub = maas_subscription_tinyllama_free.name
        models = fetch_and_assert_models_for_subscription(
            session=request_session_http,
            models_url=models_url,
            token=api_key_bound_to_free_subscription,
            expected_subscription_name=expected_sub,
            extra_headers={"x-maas-subscription": maas_subscription_tinyllama_premium.name},
        )

        LOGGER.info(
            f"[models] API key ignored X-MaaS-Subscription header — "
            f"returned {len(models)} model(s) from bound subscription '{expected_sub}'"
        )

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_access_denied_to_subscription_403(
        self,
        request_session_http: requests.Session,
        models_url: str,
        ocp_token_for_actor: str,
        maas_subscription_tinyllama_premium: MaaSSubscription,
    ) -> None:
        """Verify user token with a subscription they don't belong to gets 403."""
        headers = build_maas_headers(token=ocp_token_for_actor)
        headers["x-maas-subscription"] = maas_subscription_tinyllama_premium.name

        response = request_session_http.get(url=models_url, headers=headers, timeout=30)

        assert response.status_code == 403, (
            f"Expected 403 for inaccessible subscription, got {response.status_code}: {(response.text or '')[:200]}"
        )

        assert "application/json" in response.headers.get("Content-Type", ""), (
            f"Expected JSON response for 403, got Content-Type: {response.headers.get('Content-Type')}"
        )
        error_body = response.json()
        assert "error" in error_body, "Response missing 'error' field"
        assert error_body["error"].get("type") == "permission_error", (
            f"Expected error type 'permission_error', got {error_body['error'].get('type')}"
        )
        LOGGER.info(
            f"[models] Access denied to subscription '{maas_subscription_tinyllama_premium.name}' "
            f"-> {response.status_code} (permission_error)"
        )

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_single_subscription_auto_select(
        self,
        request_session_http: requests.Session,
        models_url: str,
        api_key_bound_to_free_subscription: str,
        maas_subscription_tinyllama_free: MaaSSubscription,
    ) -> None:
        """Verify single subscription auto-selects without needing a header."""
        expected_sub = maas_subscription_tinyllama_free.name
        models = fetch_and_assert_models_for_subscription(
            session=request_session_http,
            models_url=models_url,
            token=api_key_bound_to_free_subscription,
            expected_subscription_name=expected_sub,
        )

        LOGGER.info(f"[models] Single subscription auto-select — returned {len(models)} model(s) from '{expected_sub}'")
