from __future__ import annotations

import pytest
import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.service_account import ServiceAccount
from simple_logger.logger import get_logger

from tests.model_serving.model_server.maas_billing.maas_subscription.utils import (
    chat_payload_for_url,
    create_maas_subscription,
    poll_expected_status,
)
from tests.model_serving.model_server.maas_billing.utils import build_maas_headers
from utilities.infra import create_inference_token, login_with_user_password
from utilities.resources.maa_s_auth_policy import MaaSAuthPolicy

LOGGER = get_logger(name=__name__)

MAAS_SUBSCRIPTION_HEADER = "x-maas-subscription"


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "maas_inference_service_tinyllama_free",
    "maas_model_tinyllama_free",
    "maas_auth_policy_tinyllama_free",
    "maas_subscription_tinyllama_free",
)
class TestMultipleSubscriptionsPerModel:
    """
    Validates behavior when multiple subscriptions exist for the same model.
    """

    @pytest.mark.smoke
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_user_in_one_of_two_subscriptions_can_access_model(
        self,
        request_session_http: requests.Session,
        admin_client: DynamicClient,
        maas_free_group: str,
        maas_model_tinyllama_free,
        model_url_tinyllama_free: str,
        maas_subscription_tinyllama_free,
        maas_headers_for_actor_api_key: dict[str, str],
    ) -> None:
        """
        Create a second subscription for a different group the user is NOT in.
        User should still get 200 when explicitly selecting the correct subscription.
        """
        assert maas_free_group, "maas_free_group fixture returned empty group name"

        with create_maas_subscription(
            admin_client=admin_client,
            subscription_name="extra-subscription",
            owner_group_name="nonexistent-group-xyz",
            model_name=maas_model_tinyllama_free.name,
            tokens_per_minute=999,
            window="1m",
            priority=0,
            teardown=True,
            wait_for_resource=True,
        ) as extra_subscription:
            extra_subscription.wait_for_condition(condition="Ready", status="True", timeout=300)

            payload = chat_payload_for_url(model_url=model_url_tinyllama_free)
            explicit_headers = dict(maas_headers_for_actor_api_key)
            explicit_headers[MAAS_SUBSCRIPTION_HEADER] = maas_subscription_tinyllama_free.name

            LOGGER.info(
                "Polling for 200 with explicit subscription selection: "
                f"subscription={maas_subscription_tinyllama_free.name}"
            )

            response = poll_expected_status(
                request_session_http=request_session_http,
                model_url=model_url_tinyllama_free,
                headers=explicit_headers,
                payload=payload,
                expected_statuses={200},
            )

            assert response.status_code == 200, (
                f"Expected 200 after adding second subscription, got {response.status_code}: "
                f"{(response.text or '')[:200]}"
            )

    @pytest.mark.smoke
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_high_priority_subscription_allows_access_when_explicitly_selected(
        self,
        request_session_http: requests.Session,
        admin_client: DynamicClient,
        maas_free_group: str,
        maas_model_tinyllama_free,
        model_url_tinyllama_free: str,
        maas_subscription_tinyllama_free,
        maas_headers_for_actor_api_key: dict[str, str],
    ) -> None:
        """
        Create a second (higher priority) subscription for the same group + model.
        User should get 200 when explicitly selecting the high-priority subscription.
        """
        assert maas_free_group, "maas_free_group fixture returned empty group name"
        _ = maas_subscription_tinyllama_free

        with create_maas_subscription(
            admin_client=admin_client,
            subscription_name="high-tier-subscription",
            owner_group_name=maas_free_group,
            model_name=maas_model_tinyllama_free.name,
            tokens_per_minute=9999,
            window="1m",
            priority=10,
            teardown=True,
            wait_for_resource=True,
        ) as high_tier_subscription:
            high_tier_subscription.wait_for_condition(condition="Ready", status="True", timeout=300)

            payload = chat_payload_for_url(model_url=model_url_tinyllama_free)

            explicit_headers = dict(maas_headers_for_actor_api_key)
            explicit_headers[MAAS_SUBSCRIPTION_HEADER] = high_tier_subscription.name

            response = poll_expected_status(
                request_session_http=request_session_http,
                model_url=model_url_tinyllama_free,
                headers=explicit_headers,
                payload=payload,
                expected_statuses={200},
            )

            assert response.status_code == 200, (
                f"Expected 200 when selecting high-priority subscription '{high_tier_subscription.name}', "
                f"got {response.status_code}: {(response.text or '')[:200]}"
            )

    @pytest.mark.smoke
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
                namespace=applications_namespace,
                model_refs=[maas_model_tinyllama_free.name],
                subjects={"groups": [{"name": f"system:serviceaccounts:{applications_namespace}"}]},
                teardown=True,
                wait_for_resource=True,
            ) as service_account_auth_policy,
            create_maas_subscription(
                admin_client=admin_client,
                subscription_name="premium-subscription",
                owner_group_name=maas_premium_group,
                model_name=maas_model_tinyllama_free.name,
                tokens_per_minute=500,
                window="1m",
                priority=0,
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
            headers = build_maas_headers(token=service_account_token)
            headers[MAAS_SUBSCRIPTION_HEADER] = premium_subscription.name

            payload = chat_payload_for_url(model_url=model_url_tinyllama_free)

            response = poll_expected_status(
                request_session_http=request_session_http,
                model_url=model_url_tinyllama_free,
                headers=headers,
                payload=payload,
                expected_statuses={403, 429},
            )

            assert response.status_code in {403, 429}, (
                f"Expected 403/429 when service account selects a subscription it doesn't belong to, "
                f"got {response.status_code}: {(response.text or '')[:200]}"
            )
