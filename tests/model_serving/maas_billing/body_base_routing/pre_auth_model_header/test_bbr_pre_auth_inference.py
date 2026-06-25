"""Tests verifying BBR pre-auth inference: model name extracted from request body."""

from typing import Any, Self

import pytest
import requests
import structlog

from tests.model_serving.maas_billing.body_base_routing.pre_auth_model_header.utils import (
    assert_bbr_inference_status,
)
from tests.model_serving.maas_billing.maas_api_key.utils import search_active_api_keys
from tests.model_serving.maas_billing.utils import build_maas_headers, get_maas_models_response

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "maas_free_group",
    "maas_model_tinyllama_free",
    "maas_auth_policy_tinyllama_free",
    "maas_subscription_tinyllama_free",
    "maas_inference_service_tinyllama_free",
)
class TestBBRPreAuthInference:
    """Tests verifying that BBR pre-auth ext_proc routes inference correctly."""

    @pytest.mark.smoke
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_inference_model_in_body_only_returns_200(
        self: Self,
        request_session_http: requests.Session,
        bbr_inference_url: str,
        bbr_chat_payload: dict[str, Any],
        bbr_api_key_headers: dict[str, str],
    ) -> None:
        """Verify inference succeeds with 200; BBR pre-auth ext_proc extracts model from request body for auth."""
        assert_bbr_inference_status(
            session=request_session_http,
            inference_url=bbr_inference_url,
            headers=bbr_api_key_headers,
            payload=bbr_chat_payload,
            expected_status=200,
        )

    @pytest.mark.tier1
    def test_inference_model_in_body_only_invalid_key_returns_401(
        self: Self,
        request_session_http: requests.Session,
        bbr_inference_url: str,
        bbr_chat_payload: dict[str, Any],
    ) -> None:
        """Verify the gateway rejects inference with 401 when an invalid API key is used on the BBR endpoint."""
        assert_bbr_inference_status(
            session=request_session_http,
            inference_url=bbr_inference_url,
            headers=build_maas_headers(token="invalid-key-that-does-not-exist"),
            payload=bbr_chat_payload,
            expected_status=401,
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_list_models_returns_200_with_api_key(
        self: Self,
        request_session_http: requests.Session,
        base_url: str,
        bbr_api_key_headers: dict[str, str],
    ) -> None:
        """Verify GET /v1/models returns 200 with a valid API key after BBR deployment."""
        get_maas_models_response(
            session=request_session_http,
            base_url=base_url,
            headers=bbr_api_key_headers,
        )
        LOGGER.info("GET /v1/models with API key returned 200")

    @pytest.mark.smoke
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_maas_api_endpoint_not_broken_by_bbr(
        self: Self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
    ) -> None:
        """Verify the maas-api /v1/api-keys/search endpoint returns 200 and is not broken by the bbr-pre ext_proc."""
        search_active_api_keys(
            request_session_http=request_session_http,
            base_url=base_url,
            ocp_user_token=ocp_token_for_actor,
        )
        LOGGER.info("/v1/api-keys/search returned 200 — bbr-pre not interfering")
