from __future__ import annotations

import pytest
import requests
import structlog
from ocp_resources.maas_model_ref import MaaSModelRef
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
    "maas_model_tinyllama_premium",
    "maas_auth_policy_tinyllama_premium",
    "maas_subscription_tinyllama_premium",
)
class TestListSubscriptionsForModel:
    """Verify only subscriptions belonging to a given model are returned."""

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_returns_only_subscriptions_for_requested_model(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_headers_for_actor: dict[str, str],
        maas_model_tinyllama_free: MaaSModelRef,
        maas_subscription_tinyllama_free: MaaSSubscription,
        maas_subscription_tinyllama_premium: MaaSSubscription,
    ) -> None:
        """Verify endpoint returns only subscriptions referencing the requested model."""
        model_name = maas_model_tinyllama_free.name

        response = request_session_http.get(
            url=f"{base_url}/v1/model/{model_name}/subscriptions", headers=ocp_headers_for_actor, timeout=30
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {(response.text or '')[:200]}"

        subscriptions = response.json()
        assert isinstance(subscriptions, list), f"Expected array response, got {type(subscriptions).__name__}"

        subscription_ids = [subscription["subscription_id_header"] for subscription in subscriptions]

        assert maas_subscription_tinyllama_free.name in subscription_ids, (
            f"Expected '{maas_subscription_tinyllama_free.name}' in results for "
            f"model '{model_name}', got {subscription_ids}"
        )
        assert maas_subscription_tinyllama_premium.name not in subscription_ids, (
            f"'{maas_subscription_tinyllama_premium.name}' should not appear in results "
            f"for model '{model_name}', got {subscription_ids}"
        )

        for subscription in subscriptions:
            assert_subscription_info_schema(subscription=subscription)

        LOGGER.info(
            f"[subscriptions] GET /v1/model/{model_name}/subscriptions "
            f"-> {len(subscriptions)} subscription(s): {subscription_ids}"
        )

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_unknown_model_returns_empty_list(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_headers_for_actor: dict[str, str],
    ) -> None:
        """Verify GET /v1/model/{unknown-model}/subscriptions returns 200 with an empty list."""
        unknown_model = "nonexistent-model-xyz"

        response = request_session_http.get(
            url=f"{base_url}/v1/model/{unknown_model}/subscriptions", headers=ocp_headers_for_actor, timeout=30
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {(response.text or '')[:200]}"

        subscriptions = response.json()
        assert isinstance(subscriptions, list), f"Expected array response, got {type(subscriptions).__name__}"
        assert len(subscriptions) == 0, (
            f"Expected empty list for unknown model, got {len(subscriptions)}: {subscriptions}"
        )
        LOGGER.info(f"[subscriptions] GET /v1/model/{unknown_model}/subscriptions -> [] (empty list)")

    @pytest.mark.tier1
    def test_unauthenticated_returns_401(
        self,
        request_session_http: requests.Session,
        base_url: str,
        maas_model_tinyllama_free: MaaSModelRef,
    ) -> None:
        """Verify request without auth header returns 401."""
        model_name = maas_model_tinyllama_free.name

        response = request_session_http.get(url=f"{base_url}/v1/model/{model_name}/subscriptions", timeout=30)

        assert response.status_code == 401, f"Expected 401, got {response.status_code}: {(response.text or '')[:200]}"
        LOGGER.info(f"[subscriptions] GET /v1/model/{model_name}/subscriptions (no auth) -> {response.status_code}")
