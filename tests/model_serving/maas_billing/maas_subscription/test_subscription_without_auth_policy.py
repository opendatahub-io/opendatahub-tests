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
)
class TestSubscriptionWithoutAuthPolicy:
    """
    Validates that holding a subscription is not sufficient for model access;
    the token must also be listed in a MaaSAuthPolicy for the model.
    """

    @pytest.mark.smoke
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_subscription_without_auth_policy_gets_403(
        self,
        request_session_http: requests.Session,
        admin_client: DynamicClient,
        model_url_tinyllama_premium: str,
        maas_headers_for_actor_api_key: dict[str, str],
        maas_model_tinyllama_premium,
        maas_subscription_tinyllama_premium,
    ) -> None:
        """
        Verify that a token with a valid subscription but NOT listed in any MaaSAuthPolicy
        for that model is denied with 403.

        Given a premium model whose MaaSAuthPolicy only permits the premium group,
        When a free actor (system:authenticated but NOT premium group) is given a
          subscription for that model,
        Then the request should be denied with 403 because the AuthPolicy check
          fails regardless of subscription ownership.
        """
        with create_maas_subscription(
            admin_client=admin_client,
            subscription_namespace=maas_subscription_tinyllama_premium.namespace,
            subscription_name="e2e-free-actor-premium-sub",
            owner_group_name="system:authenticated",
            model_name=maas_model_tinyllama_premium.name,
            model_namespace=maas_model_tinyllama_premium.namespace,
            tokens_per_minute=100,
            window="1m",
            priority=0,
            teardown=True,
            wait_for_resource=True,
        ) as sub_for_free_actor:
            sub_for_free_actor.wait_for_condition(condition="Ready", status="True", timeout=300)

            headers = dict(maas_headers_for_actor_api_key)
            headers[MAAS_SUBSCRIPTION_HEADER] = sub_for_free_actor.name

            payload = chat_payload_for_url(model_url=model_url_tinyllama_premium)

            LOGGER.info(
                f"Testing: free actor has subscription '{sub_for_free_actor.name}' "
                f"but is NOT in premium MaaSAuthPolicy — expecting 403"
            )

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
