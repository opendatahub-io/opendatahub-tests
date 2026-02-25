from __future__ import annotations

import pytest
import requests
from simple_logger.logger import get_logger
from utilities.plugins.constant import RestHeader

from tests.model_serving.model_server.maas_billing.utils import build_maas_headers
from tests.model_serving.model_server.maas_billing.maas_subscription.utils import chat_payload_for_url

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
        headers = build_maas_headers(token=ocp_token_for_actor)
        payload = chat_payload_for_url(model_url=model_url_tinyllama_free)

        resp = request_session_http.post(
            url=model_url_tinyllama_free,
            headers=headers,
            json=payload,
            timeout=60,
        )
        LOGGER.info(f"test_authorized_user_gets_200 -> POST {model_url_tinyllama_free} returned {resp.status_code}")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:200]}"

    @pytest.mark.sanity
    def test_no_auth_header_gets_401(
        self,
        request_session_http: requests.Session,
        model_url_tinyllama_free: str,
    ) -> None:
        payload = chat_payload_for_url(model_url=model_url_tinyllama_free)

        resp = request_session_http.post(
            url=model_url_tinyllama_free,
            headers=RestHeader.HEADERS,
            json=payload,
            timeout=60,
        )
        LOGGER.info(f"test_no_auth_header_gets_401 -> POST {model_url_tinyllama_free} returned {resp.status_code}")
        assert resp.status_code == 401, f"Expected 401, got {resp.status_code}: {resp.text[:200]}"

    @pytest.mark.sanity
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
        assert resp.status_code == 401, f"Expected 401, got {resp.status_code}: {resp.text[:200]}"

    @pytest.mark.sanity
    def test_wrong_group_sa_denied_on_premium_model(
        self,
        request_session_http: requests.Session,
        model_url_tinyllama_premium: str,
        maas_headers_for_wrong_group_sa: dict,
    ) -> None:
        payload = chat_payload_for_url(model_url=model_url_tinyllama_premium)

        resp = request_session_http.post(
            url=model_url_tinyllama_premium,
            headers=maas_headers_for_wrong_group_sa,
            json=payload,
            timeout=60,
        )
        LOGGER.info(
            "test_wrong_group_sa_denied_on_premium_model -> "
            f"POST {model_url_tinyllama_premium} returned {resp.status_code}"
        )
        assert resp.status_code == 403, f"Expected 403, got {resp.status_code}: {resp.text[:200]}"
