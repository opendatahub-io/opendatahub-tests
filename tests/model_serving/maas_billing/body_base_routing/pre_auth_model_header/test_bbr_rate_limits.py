"""Tests verifying token rate limiting is enforced on BBR /llm/ inference paths."""

from typing import Any, Self

import pytest
import requests
import structlog

from tests.model_serving.maas_billing.utils import assert_mixed_200_and_429, build_maas_headers

LOGGER = structlog.get_logger(name=__name__)

BBR_RATE_LIMIT_MAX_REQUESTS: int = 8


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "maas_free_group",
    "maas_model_tinyllama_free",
    "maas_auth_policy_tinyllama_free",
    "maas_inference_service_tinyllama_free",
)
@pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
class TestBBRRateLimits:
    """Verify token rate limiting is enforced on BBR /llm/ inference paths."""

    @pytest.mark.tier2
    def test_bbr_inference_path_rate_limited(
        self: Self,
        request_session_http: requests.Session,
        bbr_inference_url: str,
        bbr_rate_limited_api_key: str,
        bbr_rate_limit_chat_payload: dict[str, Any],
    ) -> None:
        """Verify token rate limits are enforced on BBR /llm/ inference paths.

        Given a subscription with a low token-per-minute limit,
        when inference requests are burst-sent with a valid API key,
        then the gateway returns 429 after the quota is exhausted.
        """
        headers = build_maas_headers(token=bbr_rate_limited_api_key)
        status_codes: list[int] = []
        for attempt_idx in range(BBR_RATE_LIMIT_MAX_REQUESTS):
            response = request_session_http.post(
                url=bbr_inference_url,
                headers=headers,
                json=bbr_rate_limit_chat_payload,
                timeout=60,
            )
            status_codes.append(response.status_code)
            LOGGER.info(
                f"BBR rate limit attempt {attempt_idx + 1}/{BBR_RATE_LIMIT_MAX_REQUESTS}: status={response.status_code}"
            )
            if response.status_code == 429:
                break
        assert_mixed_200_and_429(
            actor_label="bbr-rate-limit",
            status_codes_list=status_codes,
            context="BBR /llm/ inference rate limiting",
            require_429=True,
        )
