from __future__ import annotations

import pytest
import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.service_account import ServiceAccount
from pytest_testconfig import config as py_config
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler

from tests.model_serving.model_server.maas_billing.maas_subscription.utils import chat_payload_for_url
from tests.model_serving.model_server.maas_billing.utils import build_maas_headers
from utilities.infra import create_inference_token, login_with_user_password
from utilities.resources.maa_s_auth_policy import MaaSAuthPolicy

LOGGER = get_logger(name=__name__)

MAAS_SUBSCRIPTION_HEADER = "x-maas-subscription"
INVALID_SUBSCRIPTION = "does-not-exist"


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "maas_inference_service_tinyllama_premium",
    "maas_model_tinyllama_premium",
    "maas_auth_policy_tinyllama_premium",
    "maas_subscription_tinyllama_premium",
)
class TestSubscriptionEnforcementTinyLlama:
    """Tests that MaaSSubscription correctly enforces subscription selection & limits."""

    @pytest.mark.sanity
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "premium"}], indirect=True)
    def test_subscribed_user_gets_200(
        self,
        request_session_http: requests.Session,
        model_url_tinyllama_premium: str,
        ocp_token_for_actor: str,
    ) -> None:
        resp = request_session_http.post(
            url=model_url_tinyllama_premium,
            headers=build_maas_headers(token=ocp_token_for_actor),
            json=chat_payload_for_url(model_url=model_url_tinyllama_premium),
            timeout=60,
        )
        LOGGER.info(f"test_subscribed_user_gets_200 -> {resp.status_code}")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:200]}"

    @pytest.mark.sanity
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "premium"}], indirect=True)
    def test_explicit_subscription_header_works(
        self,
        request_session_http: requests.Session,
        model_url_tinyllama_premium: str,
        ocp_token_for_actor: str,
        maas_subscription_tinyllama_premium,
    ) -> None:
        """
        - Send valid x-maas-subscription
        - Expect 200
        """
        headers = build_maas_headers(token=ocp_token_for_actor)
        headers[MAAS_SUBSCRIPTION_HEADER] = maas_subscription_tinyllama_premium.name

        resp = request_session_http.post(
            url=model_url_tinyllama_premium,
            headers=headers,
            json=chat_payload_for_url(model_url=model_url_tinyllama_premium),
            timeout=60,
        )
        LOGGER.info(f"test_explicit_subscription_header_works -> {resp.status_code}")

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:200]}"

    @pytest.mark.sanity
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "premium"}], indirect=True)
    def test_invalid_subscription_header_gets_429(
        self,
        request_session_http: requests.Session,
        model_url_tinyllama_premium: str,
        ocp_token_for_actor: str,
    ) -> None:
        """
        - Send invalid x-maas-subscription
        - Expect 429 or 403
        """
        headers = build_maas_headers(token=ocp_token_for_actor)
        headers[MAAS_SUBSCRIPTION_HEADER] = INVALID_SUBSCRIPTION

        resp = request_session_http.post(
            url=model_url_tinyllama_premium,
            headers=headers,
            json=chat_payload_for_url(model_url=model_url_tinyllama_premium),
            timeout=60,
        )
        LOGGER.info(f"test_invalid_subscription_header_gets_429 -> {resp.status_code}")

        assert resp.status_code in (429, 403), (
            f"Expected 429 or 403, got {resp.status_code}: {resp.text[:200]}"
        )
        
    @pytest.mark.sanity
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "premium"}], indirect=True)
    def test_auth_pass_no_subscription_gets_429(
        self,
        request_session_http: requests.Session,
        model_url_tinyllama_premium: str,
        ocp_token_for_actor: str,
    ) -> None:
        headers = build_maas_headers(token=ocp_token_for_actor)
        headers.pop(MAAS_SUBSCRIPTION_HEADER, None)

        payload = chat_payload_for_url(model_url=model_url_tinyllama_premium)
        last_resp: requests.Response | None = None
        for resp in TimeoutSampler(
            wait_timeout=240,
            sleep=5,
            func=request_session_http.post,
            url=model_url_tinyllama_premium,
            headers=headers,
            json=payload,
            timeout=60,
        ):
            if resp.status_code == 429:
                return

        pytest.fail(
            "Did not observe 429. "
            f"last_status={getattr(last_resp, 'status_code', None)}, "
            f"body={(getattr(last_resp, 'text', '') or '')[:200]}"
        )