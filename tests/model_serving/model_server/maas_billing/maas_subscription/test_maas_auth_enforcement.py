from __future__ import annotations

import pytest
import requests
from simple_logger.logger import get_logger

from tests.model_serving.model_server.maas_billing.maas_subscription.utils import chat_payload_for_url
from utilities.plugins.constant import RestHeader

LOGGER = get_logger(name=__name__)


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "maas_model_tinyllama_free",
    "maas_model_tinyllama_premium",
    "maas_auth_policy_tinyllama_free",
    "maas_auth_policy_tinyllama_premium",
    "maas_subscription_tinyllama_free",
    "maas_subscription_tinyllama_premium",
)
class TestMaaSAuthPolicyEnforcementTinyLlama:
    @pytest.mark.sanity
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_authorized_user_gets_200(
        self,
        request_session_http: requests.Session,
        model_url_tinyllama_free: str,
        ocp_token_for_actor: str,
    ) -> None:
        headers = {"Authorization": f"Bearer {ocp_token_for_actor}", **RestHeader.HEADERS}
        payload = chat_payload_for_url(model_url_tinyllama_free)

        r = request_session_http.post(model_url_tinyllama_free, headers=headers, json=payload, timeout=60)
        LOGGER.info(f"test_authorized_user_gets_200 -> POST {model_url_tinyllama_free} returned {r.status_code}")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:200]}"

    @pytest.mark.sanity
    def test_no_auth_header_gets_401(
        self,
        request_session_http: requests.Session,
        model_url_tinyllama_free: str,
    ) -> None:
        payload = chat_payload_for_url(model_url_tinyllama_free)

        r = request_session_http.post(model_url_tinyllama_free, headers=RestHeader.HEADERS, json=payload, timeout=60)
        LOGGER.info(f"test_no_auth_header_gets_401 -> POST {model_url_tinyllama_free} returned {r.status_code}")
        assert r.status_code == 401, f"Expected 401, got {r.status_code}: {r.text[:200]}"

    @pytest.mark.sanity
    def test_invalid_token_gets_401(
        self,
        request_session_http: requests.Session,
        model_url_tinyllama_free: str,
    ) -> None:
        headers = {"Authorization": "Bearer totally-invalid-garbage-token", **RestHeader.HEADERS}
        payload = chat_payload_for_url(model_url_tinyllama_free)

        r = request_session_http.post(model_url_tinyllama_free, headers=headers, json=payload, timeout=60)
        LOGGER.info(f"test_invalid_token_gets_401 -> POST {model_url_tinyllama_free} returned {r.status_code}")
        assert r.status_code == 401, f"Expected 401, got {r.status_code}: {r.text[:200]}"

    @pytest.mark.sanity
    def test_wrong_group_sa_denied_on_premium_model(
        self,
        request_session_http: requests.Session,
        model_url_tinyllama_premium: str,
        maas_headers_for_wrong_group_sa: dict,
    ) -> None:
        payload = chat_payload_for_url(model_url_tinyllama_premium)

        r = request_session_http.post(
            model_url_tinyllama_premium,
            headers=maas_headers_for_wrong_group_sa,
            json=payload,
            timeout=60,
        )
        LOGGER.info(
            f"test_wrong_group_sa_denied_on_premium_model -> POST {model_url_tinyllama_premium} returned {r.status_code}"
        )
        assert r.status_code == 403, f"Expected 403, got {r.status_code}: {r.text[:200]}"
