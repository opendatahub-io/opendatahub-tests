from __future__ import annotations

import pytest
import requests
import structlog

from tests.model_serving.maas_billing.maas_subscription.utils import (
    chat_payload_for_url,
)
from tests.model_serving.maas_billing.utils import build_maas_headers

LOGGER = structlog.get_logger(name=__name__)

MAAS_SUBSCRIPTION_HEADER = "x-maas-subscription"
INVALID_SUBSCRIPTION = "does-not-exist"


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "maas_inference_service_tinyllama_premium",
    "maas_model_tinyllama_premium",
    "maas_auth_policy_tinyllama_premium",
    "maas_subscription_tinyllama_premium",
)
class TestSubscriptionEnforcementTinyLlama:
    """Tests that MaaSSubscription correctly enforces subscription selection & limits."""

    @pytest.mark.smoke
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "premium"}], indirect=True)
    def test_subscribed_user_gets_200(
        self,
        request_session_http: requests.Session,
        model_url_tinyllama_premium: str,
        api_key_bound_to_premium_subscription: str,
    ) -> None:
        """
        Verify a premium user with a subscription-bound API key can access the premium model.
        """
        resp = request_session_http.post(
            url=model_url_tinyllama_premium,
            headers=build_maas_headers(token=api_key_bound_to_premium_subscription),
            json=chat_payload_for_url(model_url=model_url_tinyllama_premium),
            timeout=60,
        )
        LOGGER.info(f"test_subscribed_user_gets_200 -> {resp.status_code}")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:200]}"

    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "premium"}], indirect=True)
    def test_explicit_subscription_header_works(
        self,
        request_session_http: requests.Session,
        model_url_tinyllama_premium: str,
        maas_headers_for_actor_api_key: dict[str, str],
        maas_subscription_tinyllama_premium,
    ) -> None:
        """
        - Send valid x-maas-subscription
        - Expect 200
        """
        headers = dict(maas_headers_for_actor_api_key)
        headers[MAAS_SUBSCRIPTION_HEADER] = maas_subscription_tinyllama_premium.name

        resp = request_session_http.post(
            url=model_url_tinyllama_premium,
            headers=headers,
            json=chat_payload_for_url(model_url=model_url_tinyllama_premium),
            timeout=60,
        )
        LOGGER.info(f"test_explicit_subscription_header_works -> {resp.status_code}")

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:200]}"

    @pytest.mark.smoke
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "premium"}], indirect=True)
    def test_invalid_subscription_header_gets_429(
        self,
        request_session_http: requests.Session,
        model_url_tinyllama_premium: str,
        maas_headers_for_actor_api_key: dict[str, str],
    ) -> None:
        """
        - Send invalid x-maas-subscription
        - Expect 429 or 403
        """
        headers = dict(maas_headers_for_actor_api_key)
        headers[MAAS_SUBSCRIPTION_HEADER] = INVALID_SUBSCRIPTION

        resp = request_session_http.post(
            url=model_url_tinyllama_premium,
            headers=headers,
            json=chat_payload_for_url(model_url=model_url_tinyllama_premium),
            timeout=60,
        )

        assert resp.status_code in (429, 403), f"Expected 429 or 403, got {resp.status_code}: {resp.text[:200]}"
