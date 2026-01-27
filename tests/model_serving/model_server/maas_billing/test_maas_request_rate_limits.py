from typing import List
import pytest
from simple_logger.logger import get_logger
from tests.model_serving.model_server.maas_billing.utils import (
    assert_mixed_200_and_429,
)

LOGGER = get_logger(name=__name__)

REQUEST_RATE_MAX_REQUESTS = 10

ACTORS = [
    pytest.param({"type": "free"}, "free", id="free"),
    pytest.param({"type": "premium"}, "premium", id="premium"),
]

SCENARIO_REQUEST_RATE = {
    "id": "request-rate",
    "max_requests": REQUEST_RATE_MAX_REQUESTS,
    "max_tokens": 5,
    "sleep_between_seconds": 0.1,
    "log_prefix": "MaaS request-rate",
    "context": "request-rate burst",
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
class TestMaasRequestRateLimits:
    """
    MaaS Billing – request-rate limit tests against TinyLlama.
    """

    @pytest.fixture(scope="class")
    def scenario(self):
        return SCENARIO_REQUEST_RATE

    @pytest.mark.sanity
    def test_request_rate_limits(
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
            require_429=True,
        )
