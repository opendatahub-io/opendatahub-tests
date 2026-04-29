from __future__ import annotations

from typing import Any

import pytest
import requests
import structlog

from tests.model_serving.maas_billing.maas_api_key.utils import assert_key_rejected_at_inference
from tests.model_serving.maas_billing.utils import build_maas_headers, create_api_key, revoke_api_key
from utilities.general import generate_random_name

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "maas_model_tinyllama_free",
    "maas_auth_policy_tinyllama_free",
    "maas_subscription_tinyllama_free",
)
class TestApiKeyGatewayRejection:
    """Verify the gateway rejects invalid, revoked, and expired API keys at inference."""

    @pytest.mark.tier3
    def test_malformed_api_key_rejected(
        self,
        request_session_http: requests.Session,
        tinyllama_free_inference_url: str,
        tinyllama_free_payload: dict[str, Any],
    ) -> None:
        """Verify malformed API key is rejected with 401 at inference."""
        response = request_session_http.post(
            url=tinyllama_free_inference_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer !!!not-a-real-key-format@#$%",
            },
            json=tinyllama_free_payload,
            timeout=60,
        )
        assert response.status_code == 401, (
            f"Expected 401 for malformed API key, got {response.status_code}: {(response.text or '')[:200]}"
        )
        LOGGER.info(f"Malformed API key correctly rejected with {response.status_code}")

    @pytest.mark.tier3
    def test_empty_bearer_token_rejected(
        self,
        request_session_http: requests.Session,
        tinyllama_free_inference_url: str,
        tinyllama_free_payload: dict[str, Any],
    ) -> None:
        """Verify empty bearer token is rejected with 401 at inference."""
        response = request_session_http.post(
            url=tinyllama_free_inference_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer ",
            },
            json=tinyllama_free_payload,
            timeout=60,
        )
        assert response.status_code == 401, (
            f"Expected 401 for empty bearer token, got {response.status_code}: {(response.text or '')[:200]}"
        )
        LOGGER.info(f"Empty bearer token correctly rejected with {response.status_code}")

    @pytest.mark.tier3
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_revoked_key_rejected_at_inference(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
        tinyllama_free_inference_url: str,
        tinyllama_free_payload: dict[str, Any],
        maas_subscription_tinyllama_free,
    ) -> None:
        """Verify revoked API key is rejected with 403 at inference."""
        key_name = f"e2e-revoke-test-{generate_random_name()}"
        _, body = create_api_key(
            base_url=base_url,
            ocp_user_token=ocp_token_for_actor,
            request_session_http=request_session_http,
            api_key_name=key_name,
            subscription=maas_subscription_tinyllama_free.name,
        )
        key_id = body["id"]

        headers = build_maas_headers(token=body["key"])
        pre_check = request_session_http.post(
            url=tinyllama_free_inference_url, headers=headers, json=tinyllama_free_payload, timeout=60
        )
        assert pre_check.status_code == 200, f"Key should work before revocation, got {pre_check.status_code}"

        revoke_resp, _ = revoke_api_key(
            request_session_http=request_session_http,
            base_url=base_url,
            key_id=key_id,
            ocp_user_token=ocp_token_for_actor,
        )
        assert revoke_resp.status_code == 200, f"Failed to revoke key {key_id}: {revoke_resp.status_code}"

        assert_key_rejected_at_inference(
            request_session_http=request_session_http,
            inference_url=tinyllama_free_inference_url,
            plaintext_key=body["key"],
            payload=tinyllama_free_payload,
        )
        LOGGER.info("Revoked API key correctly rejected at inference with 403")

    @pytest.mark.tier3
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_expired_key_rejected_at_inference(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
        tinyllama_free_inference_url: str,
        tinyllama_free_payload: dict[str, Any],
        maas_subscription_tinyllama_free,
    ) -> None:
        """Verify expired API key is rejected with 403 at inference."""
        key_name = f"e2e-expired-test-{generate_random_name()}"
        _, body = create_api_key(
            base_url=base_url,
            ocp_user_token=ocp_token_for_actor,
            request_session_http=request_session_http,
            api_key_name=key_name,
            expires_in="5s",
            subscription=maas_subscription_tinyllama_free.name,
        )

        headers = build_maas_headers(token=body["key"])
        pre_check = request_session_http.post(
            url=tinyllama_free_inference_url, headers=headers, json=tinyllama_free_payload, timeout=60
        )
        assert pre_check.status_code == 200, f"Key should work before expiry, got {pre_check.status_code}"

        assert_key_rejected_at_inference(
            request_session_http=request_session_http,
            inference_url=tinyllama_free_inference_url,
            plaintext_key=body["key"],
            payload=tinyllama_free_payload,
        )
        LOGGER.info("Expired API key correctly rejected at inference with 403")
