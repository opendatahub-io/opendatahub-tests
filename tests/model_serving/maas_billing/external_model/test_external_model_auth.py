from __future__ import annotations

import pytest
import requests
import structlog

from tests.model_serving.maas_billing.external_model.utils import EXTERNAL_MODEL_NAME

LOGGER = structlog.get_logger(name=__name__)

CHAT_PAYLOAD = {
    "model": EXTERNAL_MODEL_NAME,
    "messages": [{"role": "user", "content": "hello"}],
}


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "external_model_cr",
    "external_model_ref",
    "external_model_auth_policy",
    "external_model_subscription",
)
class TestExternalModelAuth:
    """Verify auth enforcement for external model routes on the MaaS gateway."""

    @pytest.mark.tier1
    def test_invalid_key_returns_forbidden(
        self,
        request_session_http: requests.Session,
        external_model_inference_url: str,
    ) -> None:
        """Given an invalid API key, when a chat request is sent, then the gateway returns 401 or 403."""
        response = request_session_http.post(
            url=external_model_inference_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer INVALID-KEY-12345",
            },
            json=CHAT_PAYLOAD,
            timeout=60,
        )
        assert response.status_code in (401, 403), (
            f"Expected 401/403 for invalid API key, got {response.status_code}: {(response.text or '')[:200]}"
        )
        LOGGER.info(f"Invalid API key correctly rejected with {response.status_code}")

    @pytest.mark.tier1
    def test_no_key_returns_forbidden(
        self,
        request_session_http: requests.Session,
        external_model_inference_url: str,
    ) -> None:
        """Given no API key or auth header, when a chat request is sent, then the gateway returns 401 or 403."""
        response = request_session_http.post(
            url=external_model_inference_url,
            headers={"Content-Type": "application/json"},
            json=CHAT_PAYLOAD,
            timeout=60,
        )
        assert response.status_code in (401, 403), (
            f"Expected 401/403 for missing auth, got {response.status_code}: {(response.text or '')[:200]}"
        )
        LOGGER.info(f"Missing auth correctly rejected with {response.status_code}")
