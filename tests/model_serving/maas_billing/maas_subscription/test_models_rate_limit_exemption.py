from __future__ import annotations

import pytest
import requests
import structlog

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
class TestModelsRateLimitExemption:
    """Verify /v1/models is exempt from token rate limiting."""

    @pytest.mark.tier2
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_central_models_endpoint_exempt_from_rate_limiting(
        self,
        request_session_http: requests.Session,
        models_url: str,
        api_key_bound_to_free_subscription: str,
        exhausted_token_quota: None,
    ) -> None:
        """Verify /v1/models remains accessible when token quota is exhausted."""
        headers = build_maas_headers(token=api_key_bound_to_free_subscription)
        response = request_session_http.get(url=models_url, headers=headers, timeout=30)

        assert response.status_code == 200, (
            f"Expected 200 for /v1/models even when quota exhausted, "
            f"got {response.status_code}: {(response.text or '')[:200]}"
        )

        data = response.json()
        assert "data" in data, "Response missing 'data' field"
        assert isinstance(data["data"], list), "'data' must be a list"

        LOGGER.info(
            f"[models] /v1/models exempt from rate limiting -> {response.status_code} "
            f"with {len(data['data'])} model(s) (inference blocked with 429)"
        )
