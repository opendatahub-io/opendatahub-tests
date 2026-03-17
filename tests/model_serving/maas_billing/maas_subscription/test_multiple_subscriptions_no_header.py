from __future__ import annotations

import pytest
import requests
from kubernetes.dynamic import DynamicClient
from simple_logger.logger import get_logger

from tests.model_serving.maas_billing.maas_subscription.utils import (
    chat_payload_for_url,
    create_maas_subscription,
    poll_expected_status,
)

LOGGER = get_logger(name=__name__)


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

    @pytest.mark.smoke
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_multiple_matching_subscriptions_no_header_gets_403(
        self,
        request_session_http: requests.Session,
        admin_client: DynamicClient,
        maas_free_group: str,
        maas_model_tinyllama_free,
        model_url_tinyllama_free: str,
        maas_subscription_tinyllama_free,
        maas_headers_for_actor_api_key: dict[str, str],
        maas_subscription_namespace,
    ) -> None:
        """
        Verify that a token qualifying for multiple subscriptions receives 403
        when no x-maas-subscription header is provided.

        Given two subscriptions for the same model that the free actor qualifies for,
        when the actor sends a request without the x-maas-subscription header,
        then the request should be denied with 403 because the subscription
        selection is ambiguous.
        """
        _ = maas_subscription_tinyllama_free

        with create_maas_subscription(
            admin_client=admin_client,
            subscription_namespace=maas_subscription_namespace.name,
            subscription_name="e2e-second-free-subscription",
            owner_group_name=maas_free_group,
            model_name=maas_model_tinyllama_free.name,
            model_namespace=maas_model_tinyllama_free.namespace,
            tokens_per_minute=500,
            window="1m",
            priority=5,
            teardown=True,
            wait_for_resource=True,
        ) as second_subscription:
            second_subscription.wait_for_condition(condition="Ready", status="True", timeout=300)

            payload = chat_payload_for_url(model_url=model_url_tinyllama_free)

            LOGGER.info(
                f"Testing: free actor has two subscriptions "
                f"('{maas_subscription_tinyllama_free.name}' and '{second_subscription.name}') "
                f"with no x-maas-subscription header — expecting 403"
            )

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
