from __future__ import annotations

from typing import Any

import pytest
import requests
import structlog

from tests.ai_gateway.models_as_a_service.maas_api_key.utils import (
    INVALID_X_API_KEY,
    UNPREFIXED_X_API_KEY,
    build_bearer_and_x_api_key_headers,
    build_x_api_key_headers,
)
from tests.ai_gateway.models_as_a_service.utils import build_maas_headers, get_maas_models_response

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
@pytest.mark.usefixtures(
    "inference_external_model_crd_present",
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "maas_model_tinyllama_free",
    "maas_auth_policy_tinyllama_free",
    "maas_subscription_tinyllama_free",
    "x_api_key_trigger_credential_secret",
    "x_api_key_trigger_external_provider",
    "x_api_key_trigger_external_model",
    "x_api_key_auth_ready",
)
class TestXAPIKeyAuthentication:
    """Validate x-api-key and Bearer auth when a messages-format IPP ExternalModel exists."""

    @pytest.mark.tier1
    def test_bearer_api_key_lists_models(
        self,
        request_session_http: requests.Session,
        base_url: str,
        x_api_key_auth_ready: str,
    ) -> None:
        """Given x-api-key identity is enabled, when GET /v1/models uses Bearer sk-oai-*, then response is 200."""
        get_maas_models_response(
            session=request_session_http,
            base_url=base_url,
            headers=build_maas_headers(token=x_api_key_auth_ready),
        )
        LOGGER.info("GET /v1/models succeeded with Bearer API key while x-api-key identity is active")

    @pytest.mark.tier1
    def test_x_api_key_lists_models(
        self,
        request_session_http: requests.Session,
        base_url: str,
        x_api_key_auth_ready: str,
    ) -> None:
        """Given x-api-key identity is enabled, when GET /v1/models uses x-api-key, then response is 200."""
        get_maas_models_response(
            session=request_session_http,
            base_url=base_url,
            headers=build_x_api_key_headers(plaintext_api_key=x_api_key_auth_ready),
        )
        LOGGER.info("GET /v1/models succeeded with x-api-key header while x-api-key identity is active")

    @pytest.mark.tier1
    def test_x_api_key_authenticates_on_inference(
        self,
        request_session_http: requests.Session,
        tinyllama_free_inference_url: str,
        tinyllama_free_payload: dict[str, Any],
        x_api_key_auth_ready: str,
    ) -> None:
        """Given a valid API key, when inference uses x-api-key header, then response is 200."""
        response = request_session_http.post(
            url=tinyllama_free_inference_url,
            headers=build_x_api_key_headers(plaintext_api_key=x_api_key_auth_ready),
            json=tinyllama_free_payload,
            timeout=60,
        )
        assert response.status_code == 200, (
            f"Expected 200 with x-api-key header, got {response.status_code}: {(response.text or '')[:500]}"
        )
        LOGGER.info("Chat completions succeeded with x-api-key header")

    @pytest.mark.tier1
    def test_authorization_bearer_still_works_on_inference(
        self,
        request_session_http: requests.Session,
        tinyllama_free_inference_url: str,
        tinyllama_free_payload: dict[str, Any],
        x_api_key_auth_ready: str,
    ) -> None:
        """Given x-api-key identity is active, when inference uses Authorization Bearer, then response is 200."""
        response = request_session_http.post(
            url=tinyllama_free_inference_url,
            headers=build_maas_headers(token=x_api_key_auth_ready),
            json=tinyllama_free_payload,
            timeout=60,
        )
        assert response.status_code == 200, (
            f"Expected 200 with Authorization Bearer, got {response.status_code}: {(response.text or '')[:500]}"
        )
        LOGGER.info("Chat completions succeeded with Authorization Bearer")

    @pytest.mark.tier3
    def test_invalid_x_api_key_rejected_on_inference(
        self,
        request_session_http: requests.Session,
        ocp_token_for_actor: str,
        tinyllama_free_inference_url: str,
        tinyllama_free_payload: dict[str, Any],
    ) -> None:
        """Given x-api-key identity is active, when x-api-key value is invalid, then gateway returns 401 or 403."""
        response = request_session_http.post(
            url=tinyllama_free_inference_url,
            headers=build_x_api_key_headers(plaintext_api_key=INVALID_X_API_KEY),
            json=tinyllama_free_payload,
            timeout=60,
        )
        assert response.status_code in (401, 403), (
            f"Expected 401/403 for invalid x-api-key, got {response.status_code}: {(response.text or '')[:500]}"
        )
        LOGGER.info(f"Invalid x-api-key rejected with {response.status_code}")

    @pytest.mark.tier3
    def test_x_api_key_without_prefix_rejected_on_inference(
        self,
        request_session_http: requests.Session,
        ocp_token_for_actor: str,
        tinyllama_free_inference_url: str,
        tinyllama_free_payload: dict[str, Any],
    ) -> None:
        """Given x-api-key identity is active, when x-api-key lacks sk-oai- prefix, then gateway rejects the request."""
        response = request_session_http.post(
            url=tinyllama_free_inference_url,
            headers=build_x_api_key_headers(plaintext_api_key=UNPREFIXED_X_API_KEY),
            json=tinyllama_free_payload,
            timeout=60,
        )
        assert response.status_code in (401, 403), (
            f"Expected 401/403 for x-api-key without valid prefix, got {response.status_code}: "
            f"{(response.text or '')[:500]}"
        )
        LOGGER.info(f"Unprefixed x-api-key rejected with {response.status_code}")

    @pytest.mark.tier1
    def test_bearer_and_x_api_key_headers_together(
        self,
        request_session_http: requests.Session,
        tinyllama_free_inference_url: str,
        tinyllama_free_payload: dict[str, Any],
        x_api_key_auth_ready: str,
    ) -> None:
        """Given both Authorization Bearer and x-api-key are sent, when inference runs, then response is 200."""
        response = request_session_http.post(
            url=tinyllama_free_inference_url,
            headers=build_bearer_and_x_api_key_headers(plaintext_api_key=x_api_key_auth_ready),
            json=tinyllama_free_payload,
            timeout=60,
        )
        assert response.status_code == 200, (
            f"Expected 200 with both auth headers, got {response.status_code}: {(response.text or '')[:500]}"
        )
        LOGGER.info("Chat completions succeeded with both Bearer and x-api-key headers")
