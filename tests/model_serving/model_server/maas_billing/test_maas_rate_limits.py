from typing import Tuple, List

import pytest
from simple_logger.logger import get_logger
from tests.model_serving.model_server.maas_billing.utils import (
    assert_mixed_200_and_429,
)

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


@pytest.mark.usefixtures(
    "maas_inference_service_tinyllama",
    "maas_free_group",
    "maas_premium_group",
    "maas_gateway_rate_limits",
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
        ocp_token_for_actor: str,  # fixture: auth for this actor
        actor_label: str,
        scenario: dict,
        exercise_rate_limiter: Tuple[List[int], List[int]],  # fixture value, not callable
    ) -> None:
        """
        For each actor (free/premium) and each scenario
        (request-rate / token-rate), send a small burst of
        /v1/chat/completions calls.
        """
        _ = ocp_token_for_actor

        status_codes_list, total_tokens_seen_list = exercise_rate_limiter

        require_429 = scenario["id"] == "request-rate"

        assert_mixed_200_and_429(
            actor_label=actor_label,
            status_codes_list=status_codes_list,
            context=scenario["context"],
            require_429=require_429,
        )

        if scenario["id"] == "token-rate":
            LOGGER.info(
                f"MaaS token-rate[{actor_label}]: "
                f"final status_codes={status_codes_list}, "
                f"total_tokens_seen={total_tokens_seen_list}"
            )
