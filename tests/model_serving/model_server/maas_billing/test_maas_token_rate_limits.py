from typing import List
import pytest
from simple_logger.logger import get_logger
from tests.model_serving.model_server.maas_billing.utils import (
    assert_mixed_200_and_429,
)

LOGGER = get_logger(name=__name__)

TOKEN_RATE_MAX_REQUESTS = 8
LARGE_MAX_TOKENS = 80

ACTORS = [
    pytest.param({"type": "free"}, "free", id="free"),
    pytest.param({"type": "premium"}, "premium", id="premium"),
]

SCENARIO_TOKEN_RATE = {
    "id": "token-rate",
    "max_requests": TOKEN_RATE_MAX_REQUESTS,
    "max_tokens": LARGE_MAX_TOKENS,
    "sleep_between_seconds": 0.2,
    "log_prefix": "MaaS token-rate",
    "context": "token-rate tests",
}


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
@pytest.mark.parametrize(
    "ocp_token_for_actor, actor_label",
    ACTORS,
    indirect=["ocp_token_for_actor"],
    scope="class",
)
class TestMaasTokenRateLimits:
    """
    MaaS Billing – token-rate limit tests against TinyLlama.
    """

    @pytest.fixture(scope="class")
    def scenario(self):
        return SCENARIO_TOKEN_RATE

    @pytest.mark.sanity
    def test_token_rate_limits(
        self,
        ocp_token_for_actor: str,
        actor_label: str,
        scenario: dict,
        exercise_rate_limiter: List[int],
    ) -> None:

        _ = ocp_token_for_actor
        status_codes_list = exercise_rate_limiter

        assert_mixed_200_and_429(
            actor_label=actor_label,
            status_codes_list=status_codes_list,
            context=scenario["context"],
            require_429=False,
        )

        LOGGER.info(f"MaaS token-rate[{actor_label}]: final status_codes={status_codes_list}")
