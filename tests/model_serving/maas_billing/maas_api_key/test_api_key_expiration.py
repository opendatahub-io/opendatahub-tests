from __future__ import annotations

import pytest
import requests
import structlog

from tests.model_serving.maas_billing.maas_api_key.utils import assert_api_key_get_ok, get_api_key
from tests.model_serving.maas_billing.utils import assert_api_key_created_ok, create_api_key, revoke_api_key
from utilities.general import generate_random_name

LOGGER = structlog.get_logger(name=__name__)

MAAS_API_KEY_MAX_EXPIRATION_DAYS = 90


@pytest.mark.parametrize("ocp_token_for_actor", [{"type": "admin"}], indirect=True)
@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "minimal_subscription_for_free_user",
)
class TestAPIKeyExpiration:
    """Tests for API key expiration policy enforcement."""

    @pytest.mark.tier1
    def test_create_key_within_expiration_limit(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
    ) -> None:
        """Verify creating an API key with expiration below the limit succeeds."""
        expires_in_hours = max((MAAS_API_KEY_MAX_EXPIRATION_DAYS // 2) * 24, 24)
        key_name = f"e2e-exp-within-{generate_random_name()}"

        create_resp, create_body = create_api_key(
            base_url=base_url,
            ocp_user_token=ocp_token_for_actor,
            request_session_http=request_session_http,
            api_key_name=key_name,
            expires_in=f"{expires_in_hours}h",
            raise_on_error=False,
        )
        assert_api_key_created_ok(resp=create_resp, body=create_body, required_fields=("key", "expiresAt"))
        LOGGER.info(
            f"[expiration] Created key within limit: expires_in={expires_in_hours}h, "
            f"expiresAt={create_body.get('expiresAt')}"
        )
        revoke_api_key(
            request_session_http=request_session_http,
            base_url=base_url,
            key_id=create_body["id"],
            ocp_user_token=ocp_token_for_actor,
        )

    @pytest.mark.tier2
    def test_create_key_at_expiration_limit(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
    ) -> None:
        """Verify creating an API key with expiration exactly at the limit succeeds."""
        expires_in_hours = MAAS_API_KEY_MAX_EXPIRATION_DAYS * 24
        key_name = f"e2e-exp-at-limit-{generate_random_name()}"

        create_resp, create_body = create_api_key(
            base_url=base_url,
            ocp_user_token=ocp_token_for_actor,
            request_session_http=request_session_http,
            api_key_name=key_name,
            expires_in=f"{expires_in_hours}h",
            raise_on_error=False,
        )
        assert_api_key_created_ok(resp=create_resp, body=create_body, required_fields=("key", "expiresAt"))
        LOGGER.info(
            f"[expiration] Created key at limit: expires_in={expires_in_hours}h "
            f"({MAAS_API_KEY_MAX_EXPIRATION_DAYS} days)"
        )
        revoke_api_key(
            request_session_http=request_session_http,
            base_url=base_url,
            key_id=create_body["id"],
            ocp_user_token=ocp_token_for_actor,
        )

    @pytest.mark.tier1
    def test_create_key_exceeds_expiration_limit(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
    ) -> None:
        """Verify creating an API key with expiration beyond the limit returns 400."""
        exceeds_days = MAAS_API_KEY_MAX_EXPIRATION_DAYS * 2
        key_name = f"e2e-exp-exceeds-{generate_random_name()}"

        create_resp, _ = create_api_key(
            base_url=base_url,
            ocp_user_token=ocp_token_for_actor,
            request_session_http=request_session_http,
            api_key_name=key_name,
            expires_in=f"{exceeds_days * 24}h",
            raise_on_error=False,
        )
        assert create_resp.status_code == 400, (
            f"Expected 400 for expiration exceeding limit "
            f"({exceeds_days} days > {MAAS_API_KEY_MAX_EXPIRATION_DAYS} days limit), "
            f"got {create_resp.status_code}: {create_resp.text[:200]}"
        )
        error_text = create_resp.text.lower()
        assert "exceed" in error_text or "maximum" in error_text, (
            f"Expected error body to mention 'exceed' or 'maximum': {create_resp.text[:200]}"
        )
        LOGGER.info(
            f"[expiration] Correctly rejected key: {exceeds_days} days > {MAAS_API_KEY_MAX_EXPIRATION_DAYS} days limit"
        )

    @pytest.mark.tier1
    def test_create_key_without_expiration(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
        active_api_key_id: str,
    ) -> None:
        """Verify a key created without an expiration field has no expiresAt value."""
        get_resp, get_body = get_api_key(
            request_session_http=request_session_http,
            base_url=base_url,
            key_id=active_api_key_id,
            ocp_user_token=ocp_token_for_actor,
        )
        assert_api_key_get_ok(resp=get_resp, body=get_body, key_id=active_api_key_id)
        expires_at = get_body.get("expiresAt")
        assert expires_at is None, f"Expected no 'expiresAt' for key created without expiration, got: {expires_at!r}"
        LOGGER.info(f"[expiration] Key without expiration field: expiresAt={expires_at!r}")

    @pytest.mark.tier2
    def test_create_key_with_short_expiration(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
        short_expiration_api_key_id: str,
    ) -> None:
        """Verify a key created with a 1-hour expiration has a non-null expirationDate value."""
        get_resp, get_body = get_api_key(
            request_session_http=request_session_http,
            base_url=base_url,
            key_id=short_expiration_api_key_id,
            ocp_user_token=ocp_token_for_actor,
        )
        assert_api_key_get_ok(resp=get_resp, body=get_body, key_id=short_expiration_api_key_id)
        assert get_body.get("expirationDate"), (
            f"Expected non-null 'expirationDate' for 1h key, got: {get_body.get('expirationDate')!r}"
        )
        LOGGER.info(f"[expiration] 1h key expirationDate={get_body['expirationDate']}")
