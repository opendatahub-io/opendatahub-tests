from __future__ import annotations

import pytest
import requests
import structlog

from tests.model_serving.maas_billing.maas_subscription.utils import (
    chat_payload_for_url,
    poll_expected_status,
)
from tests.model_serving.maas_billing.utils import build_maas_headers
from utilities.plugins.constant import RestHeader

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    # "maas_api_gateway_reachable",
    "maas_model_tinyllama_free",
    "maas_model_tinyllama_premium",
    "maas_auth_policy_tinyllama_free",
    "maas_auth_policy_tinyllama_premium",
    "maas_subscription_tinyllama_free",
    "maas_subscription_tinyllama_premium",
)
class TestMaaSAuthPolicyEnforcementTinyLlama:
    @pytest.mark.smoke
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_authorized_user_gets_200(
        self,
        request_session_http: requests.Session,
        model_url_tinyllama_free: str,
        api_key_bound_to_free_subscription: str,
    ) -> None:
        """
        Verify a free user with a subscription-bound API key can access the free model.
        """
        resp = poll_expected_status(
            request_session_http=request_session_http,
            model_url=model_url_tinyllama_free,
            headers=build_maas_headers(token=api_key_bound_to_free_subscription),
            payload=chat_payload_for_url(model_url=model_url_tinyllama_free),
            expected_statuses={200},
        )
        LOGGER.info(f"test_authorized_user_gets_200 -> POST {model_url_tinyllama_free} returned {resp.status_code}")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:200]}"

    @pytest.mark.smoke
    def test_no_auth_header_gets_401(
        self,
        request_session_http: requests.Session,
        model_url_tinyllama_free: str,
    ) -> None:
        payload = chat_payload_for_url(model_url=model_url_tinyllama_free)

        resp = poll_expected_status(
            request_session_http=request_session_http,
            model_url=model_url_tinyllama_free,
            headers=RestHeader.HEADERS,
            payload=payload,
            expected_statuses={401},
        )
        LOGGER.info(f"test_no_auth_header_gets_401 -> POST {model_url_tinyllama_free} returned {resp.status_code}")
        assert resp.status_code == 401, f"Expected 401, got {resp.status_code}: {resp.text[:200]}"

    @pytest.mark.smoke
    def test_invalid_token_gets_401(
        self,
        request_session_http: requests.Session,
        model_url_tinyllama_free: str,
    ) -> None:
        headers = build_maas_headers(token="totally-invalid-garbage-token")
        payload = chat_payload_for_url(model_url=model_url_tinyllama_free)

        resp = request_session_http.post(
            url=model_url_tinyllama_free,
            headers=headers,
            json=payload,
            timeout=60,
        )

        LOGGER.info(f"test_invalid_token_gets_401 -> POST {model_url_tinyllama_free} returned {resp.status_code}")
        assert resp.status_code in (401, 403), f"Expected 401 or 403, got {resp.status_code}: {(resp.text or '')[:200]}"

    @pytest.mark.tier1
    def test_wrong_group_sa_denied_on_premium_model(
        self,
        request_session_http: requests.Session,
        model_url_tinyllama_premium: str,
        maas_headers_for_wrong_group_sa: dict,
    ) -> None:
        payload = chat_payload_for_url(model_url=model_url_tinyllama_premium)

        resp = poll_expected_status(
            request_session_http=request_session_http,
            model_url=model_url_tinyllama_premium,
            headers=maas_headers_for_wrong_group_sa,
            payload=payload,
            expected_statuses={401},
        )
        LOGGER.info(
            "test_wrong_group_sa_denied_on_premium_model -> "
            f"POST {model_url_tinyllama_premium} returned {resp.status_code}"
        )
        assert resp.status_code == 401, (
            f"Expected 401 (SA token not authenticated as MaaS user), got {resp.status_code}: {resp.text[:200]}"
        )
