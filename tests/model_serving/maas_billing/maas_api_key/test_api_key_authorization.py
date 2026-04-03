from __future__ import annotations

import pytest
import requests
import structlog

from tests.model_serving.maas_billing.maas_api_key.utils import get_api_key, list_api_keys
from tests.model_serving.maas_billing.utils import revoke_api_key

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "minimal_subscription_for_free_user",
)
class TestAPIKeyAuthorization:
    """Tests for MaaS API key admin and non-admin access control."""

    @pytest.mark.tier1
    def test_admin_manage_other_users_keys(
        self,
        request_session_http: requests.Session,
        base_url: str,
        admin_ocp_token: str,
        active_api_key_id: str,
        free_user_username: str,
    ) -> None:
        """Verify an admin can search for another user's keys by username and revoke them."""
        list_resp, list_body = list_api_keys(
            request_session_http=request_session_http,
            base_url=base_url,
            ocp_user_token=admin_ocp_token,
            filters={"username": free_user_username, "status": ["active"]},
            sort={"by": "created_at", "order": "desc"},
            pagination={"limit": 50, "offset": 0},
        )
        assert list_resp.status_code == 200, (
            f"Expected 200 on admin search by username, got {list_resp.status_code}: {list_resp.text[:200]}"
        )
        items: list[dict] = list_body.get("items") or list_body.get("data") or []
        key_ids = [item["id"] for item in items]
        assert active_api_key_id in key_ids, (
            f"Expected free user's key id={active_api_key_id} in admin search results, found ids={key_ids}"
        )
        LOGGER.info(f"[authz] Admin found {len(items)} active key(s) for the free user")

        revoke_resp, revoke_body = revoke_api_key(
            request_session_http=request_session_http,
            base_url=base_url,
            key_id=active_api_key_id,
            ocp_user_token=admin_ocp_token,
        )
        assert revoke_resp.status_code == 200, (
            f"Expected 200 on admin DELETE /v1/api-keys/{active_api_key_id}, "
            f"got {revoke_resp.status_code}: {revoke_resp.text[:200]}"
        )
        assert revoke_body.get("status") == "revoked", (
            f"Expected status='revoked' in admin revoke response, got: {revoke_body}"
        )
        LOGGER.info(f"[authz] Admin successfully revoked free user's key id={active_api_key_id}")

    @pytest.mark.tier1
    def test_non_admin_cannot_access_other_users_keys(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
        admin_active_api_key_id: str,
    ) -> None:
        """Verify a non-admin user gets 404 when accessing another user's API key."""
        get_resp, _ = get_api_key(
            request_session_http=request_session_http,
            base_url=base_url,
            key_id=admin_active_api_key_id,
            ocp_user_token=ocp_token_for_actor,
        )
        assert get_resp.status_code == 404, (
            f"Expected 404 (IDOR protection) on free user GET of admin's key id={admin_active_api_key_id}, "
            f"got {get_resp.status_code}: {get_resp.text[:200]}"
        )
        LOGGER.info(f"[authz] Free user correctly received 404 on GET of admin's key id={admin_active_api_key_id}")

        revoke_resp, _ = revoke_api_key(
            request_session_http=request_session_http,
            base_url=base_url,
            key_id=admin_active_api_key_id,
            ocp_user_token=ocp_token_for_actor,
        )
        assert revoke_resp.status_code == 404, (
            f"Expected 404 (IDOR protection) on free user DELETE of admin's key id={admin_active_api_key_id}, "
            f"got {revoke_resp.status_code}: {revoke_resp.text[:200]}"
        )
        LOGGER.info(f"[authz] Free user correctly received 404 on DELETE of admin's key id={admin_active_api_key_id}")

    @pytest.mark.tier2
    def test_non_admin_search_only_returns_own_keys(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
        active_api_key_id: str,
        admin_active_api_key_id: str,
    ) -> None:
        """Verify a non-admin user's search results contain only their own keys."""
        list_resp, list_body = list_api_keys(
            request_session_http=request_session_http,
            base_url=base_url,
            ocp_user_token=ocp_token_for_actor,
            filters={"status": ["active"]},
            sort={"by": "created_at", "order": "desc"},
            pagination={"limit": 50, "offset": 0},
        )
        assert list_resp.status_code == 200, (
            f"Expected 200 on free user search, got {list_resp.status_code}: {list_resp.text[:200]}"
        )
        items: list[dict] = list_body.get("items") or list_body.get("data") or []
        key_ids = [item["id"] for item in items]

        assert active_api_key_id in key_ids, (
            f"Expected free user's own key id={active_api_key_id} in results, found ids={key_ids}"
        )
        assert admin_active_api_key_id not in key_ids, (
            f"Admin's key id={admin_active_api_key_id} must NOT appear in free user's search results"
        )
        LOGGER.info(
            f"[authz] Free user search returned {len(items)} key(s) — own key present, admin's key correctly excluded"
        )

    @pytest.mark.tier2
    def test_non_admin_cannot_search_by_other_username(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
        admin_username: str,
    ) -> None:
        """Verify a non-admin user gets 403 when searching with another user's username as a filter."""
        list_resp, _ = list_api_keys(
            request_session_http=request_session_http,
            base_url=base_url,
            ocp_user_token=ocp_token_for_actor,
            filters={"username": admin_username, "status": ["active"]},
            sort={"by": "created_at", "order": "desc"},
            pagination={"limit": 50, "offset": 0},
        )
        assert list_resp.status_code == 403, (
            f"Expected 403 when free user searches by another user's username, "
            f"got {list_resp.status_code}: {list_resp.text[:200]}"
        )
        LOGGER.info("[authz] Free user correctly received 403 when searching by another user's username")
