from typing import Dict

import pytest
import time
import requests
from requests import Response
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)


REQUEST_RATE_MAX_REQUESTS = 10  # send full burst of 10 calls
TOKEN_RATE_MAX_REQUESTS = 3  # 3 "heavy" calls for token-rate
LARGE_MAX_TOKENS = 29  # each call asks for ~25 tokens


ACTORS = [
    pytest.param({"type": "free"}, "free", id="free"),
    pytest.param({"type": "premium"}, "premium", id="premium"),
]

SCENARIOS = [
    pytest.param(
        {
            "id": "request-rate",
            "max_requests": REQUEST_RATE_MAX_REQUESTS,
            "max_tokens": 5,
            "sleep_between_seconds": 0.1,
            "log_prefix": "MaaS request-rate",
            "context": "request-rate burst",
        },
        id="request-rate",
    ),
    pytest.param(
        {
            "id": "token-rate",
            "max_requests": TOKEN_RATE_MAX_REQUESTS,
            "max_tokens": LARGE_MAX_TOKENS,
            "sleep_between_seconds": 0.2,
            "log_prefix": "MaaS token-rate",
            "context": "token-rate tests",
        },
        id="token-rate",
    ),
]


def _assert_mixed_200_and_429(
    *,
    actor_label: str,
    status_codes_list: list[int],
    context: str,
) -> None:
    """
    Used for both:
    - request-rate tests
    - token-rate tests (current Kuadrant config produces 200 then 429s)
    """
    assert 200 in status_codes_list, f"{actor_label}: no 200 in {context} (status_codes={status_codes_list})"
    assert 429 in status_codes_list, f"{actor_label}: expected 429 in {context}, but saw {status_codes_list}"


@pytest.mark.usefixtures(
    "maas_inference_service_tinyllama",
    "maas_free_group",
    "maas_premium_group",
)
@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [
        pytest.param(
            {"name": "llm", "modelmesh-enabled": False},
            id="maas-billing-namespace",
        )
    ],
    indirect=True,
)
class TestMaasRateLimits:
    """
    MaaS Billing – request-rate and token-rate limit tests against TinyLlama.

    """

    @pytest.mark.parametrize("scenario", SCENARIOS)
    @pytest.mark.parametrize(
        "ocp_token_for_actor, actor_label",
        ACTORS,
        indirect=["ocp_token_for_actor"],
    )
    def test_rate_limits_for_actor_and_scenario(
        self,
        ocp_token_for_actor: str,
        actor_label: str,
        scenario: dict,
        request_session_http: requests.Session,
        model_url: str,
        maas_headers_for_actor: Dict[str, str],
        maas_models_response_for_actor: Response,
        exercise_rate_limiter,
    ) -> None:
        """
        For each actor (free/premium) and each scenario
        (request-rate / token-rate), send a small burst of
        /v1/chat/completions calls.

        """
        if scenario["id"] == "token-rate":
            LOGGER.info(
                "Sleeping 65s before token-rate scenario for actor=%s to let TokenRateLimitPolicy 1m window reset",
                actor_label,
            )
            time.sleep(65)  # noqa: FCN001

        status_codes_list, total_tokens_seen_list = exercise_rate_limiter(
            actor_label=actor_label,
            request_session_http=request_session_http,
            model_url=model_url,
            maas_headers_for_actor=maas_headers_for_actor,
            maas_models_response_for_actor=maas_models_response_for_actor,
            max_requests=scenario["max_requests"],
            max_tokens=scenario["max_tokens"],
            sleep_between_seconds=scenario["sleep_between_seconds"],
            log_prefix=scenario["log_prefix"],
        )

        _assert_mixed_200_and_429(
            actor_label=actor_label,
            status_codes_list=status_codes_list,
            context=scenario["context"],
        )

        if scenario["id"] == "token-rate":
            LOGGER.info(
                f"MaaS token-rate[{actor_label}]: final status_codes={status_codes_list}, "
                f"total_tokens_seen={total_tokens_seen_list}"
            )
