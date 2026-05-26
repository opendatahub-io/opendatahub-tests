from __future__ import annotations

from typing import Any

import pytest
import requests
import structlog

from tests.model_serving.maas_billing.utils import (
    create_api_key,
    get_maas_models_response,
    revoke_api_key,
    verify_chat_completions,
)
from utilities.general import generate_random_name

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "oidc_subscription_with_model",
    "oidc_auth_policy_patched",
)
class TestOIDCModelAccess:
    """Verify OIDC-minted API keys can access models and that revoked keys are rejected."""

    @pytest.mark.tier1
    def test_oidc_key_lists_models_and_runs_inference(
        self,
        request_session_http: requests.Session,
        base_url: str,
        model_url: str,
        oidc_minted_api_key: dict[str, Any],
    ) -> None:
        """Verify an OIDC-minted API key can list models and run inference."""
        api_key_plaintext: str = oidc_minted_api_key["key"]
        inference_headers = {
            "Authorization": f"Bearer {api_key_plaintext}",
            "Content-Type": "application/json",
        }

        models_response = get_maas_models_response(
            session=request_session_http,
            base_url=base_url,
            headers=inference_headers,
        )
        models_list: list[dict[str, Any]] = models_response.json().get("data", [])
        assert models_list, "Expected at least one model from /v1/models"

        verify_chat_completions(
            request_session_http=request_session_http,
            model_url=model_url,
            headers=inference_headers,
            models_list=models_list,
            prompt_text="Hello from external OIDC e2e",
            max_tokens=16,
            log_prefix="OIDC",
        )
        LOGGER.info(f"[oidc] Inference succeeded via OIDC-minted API key id={oidc_minted_api_key['id']}")

    @pytest.mark.tier1
    def test_revoked_oidc_key_rejected(
        self,
        request_session_http: requests.Session,
        base_url: str,
        oidc_revoked_api_key_plaintext: str,
    ) -> None:
        """Verify a revoked OIDC-minted API key is rejected with 401 or 403."""
        models_url = f"{base_url}/v1/models"
        response = request_session_http.get(
            url=models_url,
            headers={
                "Authorization": f"Bearer {oidc_revoked_api_key_plaintext}",
                "Content-Type": "application/json",
            },
            timeout=60,
        )

        assert response.status_code in (401, 403), (
            f"Expected 401 or 403 for revoked key, got {response.status_code}: {response.text[:200]}"
        )
        LOGGER.info(f"[oidc] Revoked OIDC key correctly rejected with {response.status_code}")

    @pytest.mark.tier2
    def test_create_revoke_double_revoke(
        self,
        request_session_http: requests.Session,
        base_url: str,
        external_oidc_token: str,
    ) -> None:
        """Verify first revoke returns 200 and second revoke returns 404."""
        key_name = f"e2e-oidc-double-revoke-{generate_random_name()}"
        _, api_key_body = create_api_key(
            base_url=base_url,
            ocp_user_token=external_oidc_token,
            request_session_http=request_session_http,
            api_key_name=key_name,
        )
        key_id = api_key_body["id"]

        first_revoke_response, _ = revoke_api_key(
            request_session_http=request_session_http,
            base_url=base_url,
            key_id=key_id,
            ocp_user_token=external_oidc_token,
        )
        assert first_revoke_response.status_code == 200, (
            f"First revoke should return 200, got {first_revoke_response.status_code}"
        )

        second_revoke_response, _ = revoke_api_key(
            request_session_http=request_session_http,
            base_url=base_url,
            key_id=key_id,
            ocp_user_token=external_oidc_token,
        )
        assert second_revoke_response.status_code == 404, (
            f"Second revoke should return 404, got {second_revoke_response.status_code}: "
            f"{second_revoke_response.text[:200]}"
        )

        LOGGER.info(f"[oidc] Double revoke verified: first=200, second=404 for key id={key_id}")
