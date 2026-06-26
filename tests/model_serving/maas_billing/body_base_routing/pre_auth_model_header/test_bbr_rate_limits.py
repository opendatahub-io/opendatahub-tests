"""Tests verifying token rate limiting is enforced on BBR /llm/ inference paths."""

from typing import Any, Self

import pytest
import requests
import structlog

from tests.model_serving.maas_billing.body_base_routing.pre_auth_model_header.utils import (
    BBR_RATE_LIMIT_TOKENS_PER_MINUTE,
)
from tests.model_serving.maas_billing.utils import build_maas_headers

LOGGER = structlog.get_logger(name=__name__)

BBR_RATE_LIMIT_MAX_REQUESTS: int = 10


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "maas_free_group",
    "maas_model_tinyllama_free",
    "maas_auth_policy_tinyllama_free",
    "bbr_low_limit_subscription",
    "maas_inference_service_tinyllama_free",
)
class TestBBRRateLimits:
    """Verify token rate limiting is enforced on BBR /llm/ inference paths."""

    @pytest.mark.tier2
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_bbr_inference_path_rate_limited(
        self: Self,
        request_session_http: requests.Session,
        bbr_inference_url: str,
        bbr_rate_limited_api_key: str,
        bbr_chat_payload: dict[str, Any],
    ) -> None:
        """Verify token rate limits are enforced on BBR /llm/ inference paths.

        Given a subscription with a low token-per-minute limit,
        when inference requests are burst-sent with a valid API key,
        then the gateway returns 429 after the quota is exhausted.
        """
        headers = build_maas_headers(token=bbr_rate_limited_api_key)
        for attempt in range(1, BBR_RATE_LIMIT_MAX_REQUESTS + 1):
            response = request_session_http.post(
                url=bbr_inference_url,
                headers=headers,
                json=bbr_chat_payload,
                timeout=60,
            )
            if response.status_code == 429:
                LOGGER.info(
                    f"Rate limit enforced after {attempt} request(s) — "
                    f"quota of {BBR_RATE_LIMIT_TOKENS_PER_MINUTE} tokens/min exhausted"
                )
                return
            assert response.status_code == 200, (
                f"Unexpected status {response.status_code} on attempt {attempt}: {(response.text or '')[:200]}"
            )
        pytest.fail(
            f"Rate limit not enforced after {BBR_RATE_LIMIT_MAX_REQUESTS} requests — "
            f"expected 429 before {BBR_RATE_LIMIT_TOKENS_PER_MINUTE} token quota exhaustion"
        )
