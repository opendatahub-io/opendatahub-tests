from __future__ import annotations

import pytest
import requests
import structlog
from ocp_resources.maas_subscription import MaaSSubscription

from tests.model_serving.maas_billing.maas_subscription.utils import assert_subscription_info_schema

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "maas_model_tinyllama_free",
    "maas_auth_policy_tinyllama_free",
    "maas_subscription_tinyllama_free",
)
class TestListSubscriptions:
    """Verify a user can list their subscriptions."""

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_authenticated_user_sees_accessible_subscriptions(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_headers_for_actor: dict[str, str],
        maas_subscription_tinyllama_free: MaaSSubscription,
    ) -> None:
        """Verify authenticated user gets their accessible subscriptions."""
        response = request_session_http.get(
            url=f"{base_url}/v1/subscriptions", headers=ocp_headers_for_actor, timeout=30
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {(response.text or '')[:200]}"

        subscriptions = response.json()
        assert isinstance(subscriptions, list), f"Expected array response, got {type(subscriptions).__name__}"
        assert len(subscriptions) >= 1, f"Expected at least 1 subscription, got {len(subscriptions)}"

        for subscription in subscriptions:
            assert_subscription_info_schema(subscription=subscription)

        subscription_ids = [subscription["subscription_id_header"] for subscription in subscriptions]
        assert maas_subscription_tinyllama_free.name in subscription_ids, (
            f"Expected '{maas_subscription_tinyllama_free.name}' in accessible subscriptions, got {subscription_ids}"
        )
        LOGGER.info(
            f"[subscriptions] GET /v1/subscriptions -> {len(subscriptions)} subscription(s): {subscription_ids}"
        )

    @pytest.mark.tier1
    def test_unauthenticated_returns_401(
        self,
        request_session_http: requests.Session,
        base_url: str,
    ) -> None:
        """Verify request without auth header returns 401."""
        response = request_session_http.get(url=f"{base_url}/v1/subscriptions", timeout=30)

        assert response.status_code == 401, f"Expected 401, got {response.status_code}: {(response.text or '')[:200]}"
        LOGGER.info(f"[subscriptions] GET /v1/subscriptions (no auth) -> {response.status_code}")

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_subscription_response_includes_model_refs_with_rate_limits(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_headers_for_actor: dict[str, str],
        maas_subscription_tinyllama_free: MaaSSubscription,
    ) -> None:
        """Verify subscription response includes model_refs with rate limit info."""
        response = request_session_http.get(
            url=f"{base_url}/v1/subscriptions", headers=ocp_headers_for_actor, timeout=30
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {(response.text or '')[:200]}"

        subscriptions = response.json()
        assert isinstance(subscriptions, list), f"Expected array response, got {type(subscriptions).__name__}"

        target_subscription = next(
            (
                subscription
                for subscription in subscriptions
                if subscription["subscription_id_header"] == maas_subscription_tinyllama_free.name
            ),
            None,
        )
        assert target_subscription is not None, (
            f"Subscription '{maas_subscription_tinyllama_free.name}' not found in "
            f"{[subscription['subscription_id_header'] for subscription in subscriptions]}"
        )

        assert len(target_subscription["model_refs"]) >= 1, "Expected at least 1 model_ref"
        model_ref = target_subscription["model_refs"][0]
        assert isinstance(model_ref["name"], str), "model_ref name must be a string"

        assert "token_rate_limits" in model_ref, "model_ref missing 'token_rate_limits'"
        assert len(model_ref["token_rate_limits"]) >= 1, "Expected at least 1 token_rate_limit"
        rate_limit = model_ref["token_rate_limits"][0]
        assert "limit" in rate_limit, "token_rate_limit missing 'limit'"
        assert "window" in rate_limit, "token_rate_limit missing 'window'"

        LOGGER.info(
            f"[subscriptions] Subscription '{target_subscription['subscription_id_header']}' "
            f"model_refs: {target_subscription['model_refs']}"
        )
