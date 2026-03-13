from __future__ import annotations

import pytest
import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.maas_auth_policy import MaaSAuthPolicy
from simple_logger.logger import get_logger

from tests.model_serving.maas_billing.maas_subscription.utils import (
    chat_payload_for_url,
    create_maas_subscription,
    poll_expected_status,
)
from utilities.general import generate_random_name

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
class TestMultipleAuthPoliciesPerModel:
    """
    Multiple auth policies for one model aggregate with OR logic.
    We validate against the PREMIUM model:
      - baseline premium auth policy requires premium group (so FREE actor should get 403)
      - add extra auth policy that allows system:authenticated
      - add matching subscription for system:authenticated
      - FREE actor should now get 200 when explicitly selecting that subscription
    """

    @pytest.mark.smoke
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_two_auth_policies_or_logic_allows_access(
        self,
        request_session_http: requests.Session,
        admin_client: DynamicClient,
        maas_model_tinyllama_premium,
        model_url_tinyllama_premium: str,
        maas_subscription_tinyllama_premium,
        maas_headers_for_actor_api_key: dict[str, str],
    ) -> None:
        """
        Validate OR behavior for multiple AuthPolicies on the PREMIUM model.
        Steps:
        1) Baseline: FREE actor is denied (403) because only the premium group is allowed.
        2) Add a second AuthPolicy allowing system:authenticated.
        3) Add a matching subscription for system:authenticated.
        4) FREE actor explicitly selects that subscription -> allowed (200).
        """

        baseline_headers = dict(maas_headers_for_actor_api_key)
        baseline_headers[MAAS_SUBSCRIPTION_HEADER] = maas_subscription_tinyllama_premium.name
        baseline_payload = chat_payload_for_url(model_url=model_url_tinyllama_premium)

        baseline_resp = poll_expected_status(
            request_session_http=request_session_http,
            model_url=model_url_tinyllama_premium,
            headers=baseline_headers,
            payload=baseline_payload,
            expected_statuses={403},
        )
        assert baseline_resp.status_code == 403, (
            f"Expected baseline 403 for FREE actor on premium model, got {baseline_resp.status_code}: "
            f"{(baseline_resp.text or '')[:200]}"
        )

        suffix = generate_random_name()
        extra_auth_policy_name = f"e2e-premium-system-auth-{suffix}"
        system_auth_sub_name = f"e2e-premium-system-auth-sub-{suffix}"

        with (
            MaaSAuthPolicy(
                client=admin_client,
                name=extra_auth_policy_name,
                namespace=maas_subscription_tinyllama_premium.namespace,
                model_refs=[
                    {
                        "name": maas_model_tinyllama_premium.name,
                        "namespace": maas_model_tinyllama_premium.namespace,
                    }
                ],
                subjects={"groups": [{"name": "system:authenticated"}]},
                teardown=True,
                wait_for_resource=True,
            ) as extra_auth_policy,
            create_maas_subscription(
                admin_client=admin_client,
                subscription_namespace=maas_subscription_tinyllama_premium.namespace,
                subscription_name=system_auth_sub_name,
                owner_group_name="system:authenticated",
                model_name=maas_model_tinyllama_premium.name,
                model_namespace=maas_model_tinyllama_premium.namespace,
                tokens_per_minute=100,
                window="1m",
                priority=0,
                teardown=True,
                wait_for_resource=True,
            ) as system_auth_subscription,
        ):
            extra_auth_policy.wait_for_condition(condition="Ready", status="True", timeout=300)
            system_auth_subscription.wait_for_condition(condition="Ready", status="True", timeout=300)
            payload = chat_payload_for_url(model_url=model_url_tinyllama_premium)
            explicit_headers = dict(maas_headers_for_actor_api_key)
            explicit_headers[MAAS_SUBSCRIPTION_HEADER] = system_auth_subscription.name

            LOGGER.info(
                "Polling for 200 on premium model with OR auth policy: "
                f"auth_policy={extra_auth_policy_name}, subscription={system_auth_subscription.name}"
            )

            response = poll_expected_status(
                request_session_http=request_session_http,
                model_url=model_url_tinyllama_premium,
                headers=explicit_headers,
                payload=payload,
                expected_statuses={200},
            )

            assert response.status_code == 200, (
                f"Expected 200 with second AuthPolicy (OR logic), got {response.status_code}: "
                f"{(response.text or '')[:200]}"
            )

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_delete_one_auth_policy_other_still_works_on_premium_model(
        self,
        request_session_http: requests.Session,
        admin_client: DynamicClient,
        maas_model_tinyllama_premium,
        model_url_tinyllama_premium: str,
        maas_subscription_tinyllama_premium,
        maas_headers_for_actor_api_key: dict[str, str],
    ) -> None:
        """
        Validate behavior when removing one of multiple AuthPolicies on PREMIUM model.
        Steps:
        1) Baseline: FREE actor is denied (403).
        2) Add extra AuthPolicy allowing system:authenticated + matching subscription.
        3) FREE actor explicitly selects that subscription -> allowed (200).
        4) Delete the extra AuthPolicy -> FREE actor denied again (403) because baseline premium policy remains.
        """
        baseline_headers = dict(maas_headers_for_actor_api_key)
        baseline_headers[MAAS_SUBSCRIPTION_HEADER] = maas_subscription_tinyllama_premium.name
        baseline_payload = chat_payload_for_url(model_url=model_url_tinyllama_premium)

        baseline_resp = poll_expected_status(
            request_session_http=request_session_http,
            model_url=model_url_tinyllama_premium,
            headers=baseline_headers,
            payload=baseline_payload,
            expected_statuses={403},
        )
        assert baseline_resp.status_code == 403, (
            f"Expected baseline 403 for FREE actor on premium model, got {baseline_resp.status_code}: "
            f"{(baseline_resp.text or '')[:200]}"
        )

        suffix = generate_random_name()
        extra_auth_policy_name = f"e2e-premium-extra-auth-{suffix}"
        system_auth_sub_name = f"e2e-premium-extra-sub-{suffix}"
        payload = chat_payload_for_url(model_url=model_url_tinyllama_premium)

        with (
            MaaSAuthPolicy(
                client=admin_client,
                name=extra_auth_policy_name,
                namespace=maas_subscription_tinyllama_premium.namespace,
                model_refs=[
                    {
                        "name": maas_model_tinyllama_premium.name,
                        "namespace": maas_model_tinyllama_premium.namespace,
                    }
                ],
                subjects={"groups": [{"name": "system:authenticated"}]},
                teardown=True,
                wait_for_resource=True,
            ) as extra_auth_policy,
            create_maas_subscription(
                admin_client=admin_client,
                subscription_namespace=maas_subscription_tinyllama_premium.namespace,
                subscription_name=system_auth_sub_name,
                owner_group_name="system:authenticated",
                model_name=maas_model_tinyllama_premium.name,
                model_namespace=maas_model_tinyllama_premium.namespace,
                tokens_per_minute=100,
                window="1m",
                priority=0,
                teardown=True,
                wait_for_resource=True,
            ) as system_auth_subscription,
        ):
            extra_auth_policy.wait_for_condition(condition="Ready", status="True", timeout=300)
            system_auth_subscription.wait_for_condition(condition="Ready", status="True", timeout=300)
            explicit_headers = dict(maas_headers_for_actor_api_key)
            explicit_headers[MAAS_SUBSCRIPTION_HEADER] = system_auth_subscription.name

            poll_expected_status(
                request_session_http=request_session_http,
                model_url=model_url_tinyllama_premium,
                headers=explicit_headers,
                payload=payload,
                expected_statuses={200},
            )

            extra_auth_policy.delete(wait=True)

            response = poll_expected_status(
                request_session_http=request_session_http,
                model_url=model_url_tinyllama_premium,
                headers=explicit_headers,
                payload=payload,
                expected_statuses={403},
            )

            assert response.status_code == 403, (
                f"Expected 403 after deleting extra AuthPolicy, got {response.status_code}: "
                f"{(response.text or '')[:200]}"
            )
