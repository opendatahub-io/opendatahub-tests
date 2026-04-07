from __future__ import annotations

import pytest
import requests
import structlog
from ocp_resources.maas_subscription import MaaSSubscription

from tests.model_serving.maas_billing.maas_subscription.utils import (
    chat_payload_for_url,
    poll_expected_status,
)

LOGGER = structlog.get_logger(name=__name__)

MAAS_SUBSCRIPTION_HEADER = "x-maas-subscription"


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "maas_inference_service_tinyllama_premium",
    "maas_model_tinyllama_premium",
    "maas_auth_policy_tinyllama_premium",
    "maas_subscription_tinyllama_premium",
    "minimal_subscription_for_free_user",
)
class TestSubscriptionWithoutAuthPolicy:
    """
    Validates that holding a subscription is not sufficient for model access;
    the token must also be listed in a MaaSAuthPolicy for the model.
    """

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_subscription_without_auth_policy_gets_403(
        self,
        request_session_http: requests.Session,
        model_url_tinyllama_premium: str,
        maas_headers_for_actor_api_key: dict[str, str],
        free_actor_premium_subscription: MaaSSubscription,
    ) -> None:
        """
        Verify that a user with a subscription but without AuthPolicy access
        to the model gets 403.

        A free user is given a subscription for the premium model,
        but since the user is not allowed by the model's MaaSAuthPolicy,
        the request should be denied with 403.
        """
        LOGGER.info(
            f"Testing: free actor has subscription '{free_actor_premium_subscription.name}' "
            f"but is NOT in premium MaaSAuthPolicy — expecting 403"
        )

        headers = dict(maas_headers_for_actor_api_key)
        headers[MAAS_SUBSCRIPTION_HEADER] = free_actor_premium_subscription.name

        payload = chat_payload_for_url(model_url=model_url_tinyllama_premium)

        response = poll_expected_status(
            request_session_http=request_session_http,
            model_url=model_url_tinyllama_premium,
            headers=headers,
            payload=payload,
            expected_statuses={403},
        )

        assert response.status_code == 403, (
            f"Expected 403 for token with subscription but not in MaaSAuthPolicy, "
            f"got {response.status_code}: {(response.text or '')[:200]}"
        )
