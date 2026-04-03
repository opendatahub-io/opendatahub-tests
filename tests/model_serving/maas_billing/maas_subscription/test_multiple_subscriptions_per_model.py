from __future__ import annotations

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.service_account import ServiceAccount

from tests.model_serving.maas_billing.maas_subscription.utils import (
    chat_payload_for_url,
    create_maas_subscription,
    poll_expected_status,
)
from tests.model_serving.maas_billing.utils import build_maas_headers
from utilities.infra import create_inference_token, login_with_user_password
from utilities.resources.maa_s_auth_policy import MaaSAuthPolicy

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_inference_service_tinyllama_free",
    "maas_model_tinyllama_free",
    "maas_auth_policy_tinyllama_free",
    "maas_subscription_tinyllama_free",
)
class TestMultipleSubscriptionsPerModel:
    """
    Validates behavior when multiple subscriptions exist for the same model.
    """

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_user_in_one_of_two_subscriptions_can_access_model(
        self,
        request_session_http: requests.Session,
        model_url_tinyllama_free: str,
        extra_subscription_with_api_key: str,
    ) -> None:
        """
        Create a second subscription for a different group the user is NOT in.
        User should still get 200 — API key is bound to the original free subscription,
        verifying OR-logic: a second subscription does not block access.
        """
        LOGGER.info("Polling for 200 — user's API key bound to original subscription despite a second one existing")

        response = poll_expected_status(
            request_session_http=request_session_http,
            model_url=model_url_tinyllama_free,
            headers=build_maas_headers(token=extra_subscription_with_api_key),
            payload=chat_payload_for_url(model_url=model_url_tinyllama_free),
            expected_statuses={200},
        )

        assert response.status_code == 200, (
            f"Expected 200 after adding second subscription, got {response.status_code}: {(response.text or '')[:200]}"
        )

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_high_priority_subscription_allows_access_when_explicitly_selected(
        self,
        request_session_http: requests.Session,
        model_url_tinyllama_free: str,
        high_tier_subscription_with_api_key: str,
    ) -> None:
        """
        Create a second (higher priority) subscription for the same group + model.
        User should get 200 — API key is minted and bound to the high-priority subscription
        at creation time.
        """
        response = poll_expected_status(
            request_session_http=request_session_http,
            model_url=model_url_tinyllama_free,
            headers=build_maas_headers(token=high_tier_subscription_with_api_key),
            payload=chat_payload_for_url(model_url=model_url_tinyllama_free),
            expected_statuses={200},
        )

        assert response.status_code == 200, (
            f"Expected 200 when using API key bound to high-priority subscription, "
            f"got {response.status_code}: {(response.text or '')[:200]}"
        )

    @pytest.mark.tier1
    def test_service_account_cannot_use_subscription_it_does_not_belong_to(
        self,
        request_session_http: requests.Session,
        admin_client: DynamicClient,
        maas_api_server_url: str,
        original_user: str,
        maas_premium_group: str,
        maas_model_tinyllama_free,
        model_url_tinyllama_free: str,
        maas_unprivileged_model_namespace,
        maas_subscription_namespace,
    ) -> None:
        """
        A service account explicitly selecting a subscription it does not belong to
        should be denied.
        """
        service_account_name = "test-service-account"

        login_ok = login_with_user_password(api_address=maas_api_server_url, user=original_user)
        assert login_ok, f"Failed to login as original_user={original_user}"

        applications_namespace = maas_unprivileged_model_namespace.name
        assert applications_namespace, "applications_namespace name is empty"

        with (
            MaaSAuthPolicy(
                client=admin_client,
                name="service-account-access-policy",
                namespace=maas_subscription_namespace.name,
                model_refs=[
                    {
                        "name": maas_model_tinyllama_free.name,
                        "namespace": maas_model_tinyllama_free.namespace,
                    }
                ],
                subjects={"groups": [{"name": f"system:serviceaccounts:{applications_namespace}"}]},
                teardown=True,
                wait_for_resource=True,
            ) as service_account_auth_policy,
            create_maas_subscription(
                admin_client=admin_client,
                subscription_namespace=maas_subscription_namespace.name,
                subscription_name="premium-subscription",
                owner_group_name=maas_premium_group,
                model_name=maas_model_tinyllama_free.name,
                model_namespace=maas_model_tinyllama_free.namespace,
                tokens_per_minute=500,
                window="1m",
                priority=2,
                teardown=True,
                wait_for_resource=True,
            ) as premium_subscription,
            ServiceAccount(
                client=admin_client,
                namespace=applications_namespace,
                name=service_account_name,
                teardown=True,
            ) as service_account,
        ):
            service_account_auth_policy.wait_for_condition(condition="Ready", status="True", timeout=300)
            premium_subscription.wait_for_condition(condition="Ready", status="True", timeout=300)
            service_account.wait(timeout=60)

            service_account_token = create_inference_token(model_service_account=service_account)
            # SA token is not a valid MaaS API key — request is rejected (401/403/429)
            headers = build_maas_headers(token=service_account_token)

            payload = chat_payload_for_url(model_url=model_url_tinyllama_free)

            response = poll_expected_status(
                request_session_http=request_session_http,
                model_url=model_url_tinyllama_free,
                headers=headers,
                payload=payload,
                expected_statuses={401, 403, 429},
            )

            assert response.status_code in {401, 403, 429}, (
                f"Expected 401/403/429 when service account token has no valid MaaS subscription, "
                f"got {response.status_code}: {(response.text or '')[:200]}"
            )
