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

pytestmark = [
    pytest.mark.usefixtures(
        "maas_unprivileged_model_namespace",
        "maas_subscription_controller_enabled_latest",
        "maas_gateway_api",
        "maas_api_gateway_reachable",
        "maas_free_group",
        "maas_model_tinyllama_free",
        "maas_auth_policy_tinyllama_free",
        "maas_subscription_tinyllama_free",
        "maas_inference_service_tinyllama_free",
    ),
]


class TestBBRPreAuthInferenceAuth:
    """Verify auth enforcement for BBR /llm/ path-routed inference on the MaaS gateway."""

    @pytest.mark.tier1
    def test_inference_invalid_key_returns_401_or_403(
        self: Self,
        request_session_http: requests.Session,
        bbr_inference_url: str,
        bbr_chat_payload: dict[str, Any],
    ) -> None:
        """Verify the gateway rejects inference with 401 or 403 when an invalid API key is used."""
        response = request_session_http.post(
            url=bbr_inference_url,
            headers=build_maas_headers(token="invalid-key-that-does-not-exist"),
            json=bbr_chat_payload,
            timeout=60,
        )
        assert response.status_code in (401, 403), (
            f"Expected 401 or 403 for invalid API key on BBR inference, got {response.status_code}"
        )
        LOGGER.info(f"BBR inference with invalid API key returned {response.status_code}")

    @pytest.mark.tier1
    def test_inference_no_api_key_returns_401_or_403(
        self: Self,
        request_session_http: requests.Session,
        bbr_inference_url: str,
        bbr_chat_payload: dict[str, Any],
    ) -> None:
        """Verify the gateway rejects inference with 401 or 403 when no Authorization header is present.

        Both codes are valid per the MaaS AuthPolicy: 401 for unauthenticated requests,
        403 for requests denied at the authorization layer before credentials are evaluated.
        """
        response = request_session_http.post(
            url=bbr_inference_url,
            headers={},
            json=bbr_chat_payload,
            timeout=60,
        )
        assert response.status_code in (401, 403), (
            f"Expected 401 or 403 for missing auth on BBR inference, got {response.status_code}"
        )
        LOGGER.info(f"BBR inference with no API key returned {response.status_code}")


@pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
class TestBBRPreAuthInference:
    """Regression tests verifying /llm/ path-routed inference is not broken after BBR pre-auth ext_proc is deployed."""

    @pytest.mark.smoke
    def test_inference_model_in_body_only_returns_200(
        self: Self,
        request_session_http: requests.Session,
        bbr_inference_url: str,
        bbr_chat_payload: dict[str, Any],
        bbr_api_key_headers: dict[str, str],
    ) -> None:
        """Verify /llm/ path-routed inference returns 200 and is not broken by the BBR pre-auth ext_proc deployment."""
        assert_bbr_inference_status(
            session=request_session_http,
            inference_url=bbr_inference_url,
            headers=bbr_api_key_headers,
            payload=bbr_chat_payload,
            expected_status=200,
        )

    @pytest.mark.tier1
    def test_inference_path_wins_over_body_model_for_auth(
        self: Self,
        request_session_http: requests.Session,
        bbr_inference_url: str,
        bbr_api_key_headers: dict[str, str],
    ) -> None:
        """Verify that on /llm/ paths, auth uses path identity — wrong body model returns 404, not 401/403.

        BBR pre-auth sets X-Gateway-Model-Name from the body, so post-auth routing returns 404
        when body.model does not match a real model. Auth must succeed (not 401/403); 404 is
        the expected routing outcome for a mismatched body model.
        """
        wrong_model_payload: dict[str, Any] = {
            "model": "wrong-model-in-body",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1,
        }
        assert_bbr_inference_status(
            session=request_session_http,
            inference_url=bbr_inference_url,
            headers=bbr_api_key_headers,
            payload=wrong_model_payload,
            expected_status=404,
        )
        LOGGER.info("Path-wins auth verified: auth passed and routing returned 404 for wrong body model")

    @pytest.mark.tier1
    def test_inference_streaming_returns_sse(
        self: Self,
        request_session_http: requests.Session,
        bbr_inference_url: str,
        bbr_chat_payload: dict[str, Any],
        bbr_api_key_headers: dict[str, str],
    ) -> None:
        """Verify that inference with stream=True returns 200 with SSE text/event-stream chunks."""
        streaming_payload = {**bbr_chat_payload, "stream": True}
        with request_session_http.post(
            url=bbr_inference_url,
            headers=bbr_api_key_headers,
            json=streaming_payload,
            timeout=60,
            stream=True,
        ) as response:
            assert response.status_code == 200, f"Expected 200 for streaming BBR inference, got {response.status_code}"
            content_type = response.headers.get("content-type", "")
            assert "text/event-stream" in content_type, (
                f"Expected text/event-stream content type for streaming response, got '{content_type}'"
            )
            chunks = [line for line, _ in zip(response.iter_lines(), range(10), strict=False) if line]
            assert any(chunk.startswith(b"data:") for chunk in chunks), (
                f"Expected SSE data: chunks in streaming response — first chunks: {chunks!r}"
            )
        LOGGER.info(f"Streaming BBR inference returned 200 with {len(chunks)} SSE chunks")

    @pytest.mark.smoke
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
