from __future__ import annotations

import pytest
import requests
import structlog

from tests.model_serving.maas_billing.maas_subscription.utils import (
    chat_payload_for_url,
    poll_expected_status,
)
from tests.model_serving.maas_billing.utils import build_maas_headers

LOGGER = structlog.get_logger(name=__name__)


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
    @pytest.mark.smoke
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_premium_model_denies_free_actor_by_default(
        self,
        request_session_http: requests.Session,
        model_url_tinyllama_premium: str,
        maas_headers_for_actor_api_key: dict[str, str],
    ) -> None:
        """
        Verify FREE actor is denied by default on the premium model.
        The API key is not bound to the premium subscription, and auth policy denies the free group.
        """
        baseline_payload = chat_payload_for_url(model_url=model_url_tinyllama_premium)

        baseline_response = poll_expected_status(
            request_session_http=request_session_http,
            model_url=model_url_tinyllama_premium,
            headers=maas_headers_for_actor_api_key,
            payload=baseline_payload,
            expected_statuses={403},
        )

        assert baseline_response.status_code == 403, (
            f"Expected baseline 403 for FREE actor on premium model, got "
            f"{baseline_response.status_code}: {(baseline_response.text or '')[:200]}"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_two_auth_policies_or_logic_allows_access(
        self,
        request_session_http: requests.Session,
        model_url_tinyllama_premium: str,
        premium_system_authenticated_access,
        api_key_bound_to_system_auth_subscription: str,
    ) -> None:
        """
        Verify FREE actor can access the premium model when an extra AuthPolicy
        and matching subscription for system:authenticated exist.
        API key is minted and bound to the system:authenticated subscription at creation time.
        """
        LOGGER.info(
            f"Polling for 200 on premium model with OR auth policy: "
            f"auth_policy={premium_system_authenticated_access['auth_policy'].name}, "
            f"subscription={premium_system_authenticated_access['subscription'].name}"
        )

        response = poll_expected_status(
            request_session_http=request_session_http,
            model_url=model_url_tinyllama_premium,
            headers=build_maas_headers(token=api_key_bound_to_system_auth_subscription),
            payload=chat_payload_for_url(model_url=model_url_tinyllama_premium),
            expected_statuses={200},
        )
        assert response.status_code == 200, (
            f"Expected 200 with second AuthPolicy (OR logic), got {response.status_code}: {(response.text or '')[:200]}"
        )

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_delete_extra_auth_policy_denies_access_on_premium_model(
        self,
        request_session_http: requests.Session,
        model_url_tinyllama_premium: str,
        premium_system_authenticated_access,
        api_key_bound_to_system_auth_subscription: str,
    ) -> None:
        """
        Verify FREE actor loses access again after the extra AuthPolicy is deleted.
        API key is minted and bound to the system:authenticated subscription at creation time.
        """
        headers = build_maas_headers(token=api_key_bound_to_system_auth_subscription)
        payload = chat_payload_for_url(model_url=model_url_tinyllama_premium)

        poll_expected_status(
            request_session_http=request_session_http,
            model_url=model_url_tinyllama_premium,
            headers=headers,
            payload=payload,
            expected_statuses={200},
        )

        premium_system_authenticated_access["auth_policy"].delete(wait=True)

        response = poll_expected_status(
            request_session_http=request_session_http,
            model_url=model_url_tinyllama_premium,
            headers=headers,
            payload=payload,
            expected_statuses={403},
        )

        assert response.status_code == 403, (
            f"Expected 403 after deleting extra AuthPolicy, got {response.status_code}: {(response.text or '')[:200]}"
        )
