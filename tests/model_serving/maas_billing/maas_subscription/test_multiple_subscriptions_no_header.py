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


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "maas_inference_service_tinyllama_free",
    "maas_model_tinyllama_free",
    "maas_auth_policy_tinyllama_free",
    "maas_subscription_tinyllama_free",
)
class TestMultipleSubscriptionsNoHeader:
    """
    Validates that a token qualifying for multiple subscriptions on the same model
    is denied when no x-maas-subscription header is provided to disambiguate.
    """

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_multiple_matching_subscriptions_no_header_gets_403(
        self,
        request_session_http: requests.Session,
        model_url_tinyllama_free: str,
        maas_headers_for_actor_api_key: dict[str, str],
        second_free_subscription: MaaSSubscription,
    ) -> None:
        """
        Verify that a token qualifying for multiple subscriptions receives 403
        when no x-maas-subscription header is provided.

        Given two subscriptions for the same model that the free actor qualifies for,
        when the actor sends a request without the x-maas-subscription header,
        then the request should be denied with 403 because the subscription
        selection is ambiguous.
        """
        LOGGER.info(
            f"Testing: free actor has two subscriptions including '{second_free_subscription.name}' "
            f"with no x-maas-subscription header — expecting 403"
        )

        payload = chat_payload_for_url(model_url=model_url_tinyllama_free)

        response = poll_expected_status(
            request_session_http=request_session_http,
            model_url=model_url_tinyllama_free,
            headers=maas_headers_for_actor_api_key,
            payload=payload,
            expected_statuses={403},
        )

        assert response.status_code == 403, (
            f"Expected 403 when multiple subscriptions exist and no header is provided, "
            f"got {response.status_code}: {(response.text or '')[:200]}"
        )
