from __future__ import annotations

import pytest
import requests
import structlog

from tests.ai_gateway.models_as_a_service.maas_api_key.utils import (
    FreeUserKeysAcrossSubscriptions,
    assert_api_keys_status,
    assert_bulk_dry_run_preview,
    assert_bulk_revoke_by_subscription_success,
    assert_bulk_revoke_success,
    bulk_revoke_api_keys,
    get_api_key,
    resolve_api_key_username,
)

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "maas_model_tinyllama_free",
    "maas_auth_policy_tinyllama_free",
    "maas_subscription_tinyllama_free",
    "minimal_subscription_for_free_user",
)
class TestAPIKeyBulkOperations:
    """Tests for MaaS API key bulk revoke operations."""

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "admin"}], indirect=True)
    def test_bulk_revoke_own_keys(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
        three_active_api_key_ids: list[str],
    ) -> None:
        """Verify a user can bulk revoke all their own active API keys."""
        username = resolve_api_key_username(
            request_session_http=request_session_http,
            base_url=base_url,
            key_id=three_active_api_key_ids[0],
            ocp_user_token=ocp_token_for_actor,
        )

        revoked_count = assert_bulk_revoke_success(
            request_session_http=request_session_http,
            base_url=base_url,
            ocp_user_token=ocp_token_for_actor,
            username=username,
            min_revoked_count=3,
        )
        LOGGER.info(f"[bulk-revoke] User {username} bulk revoked {revoked_count} key(s)")

        for key_id in three_active_api_key_ids:
            get_resp, get_body = get_api_key(
                request_session_http=request_session_http,
                base_url=base_url,
                key_id=key_id,
                ocp_user_token=ocp_token_for_actor,
            )
            assert get_resp.status_code == 200, (
                f"Expected 200 on GET /v1/api-keys/{key_id} after bulk-revoke, "
                f"got {get_resp.status_code}: {get_resp.text[:200]}"
            )
            assert get_body.get("status") == "revoked", (
                f"Expected key id={key_id} to have status='revoked', got: {get_body.get('status')}"
            )
        LOGGER.info(f"[bulk-revoke] All {len(three_active_api_key_ids)} key(s) confirmed revoked")

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_bulk_revoke_other_user_forbidden(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
    ) -> None:
        """Verify a non-admin user gets 403 when attempting to bulk revoke another user's keys."""
        bulk_resp, _ = bulk_revoke_api_keys(
            request_session_http=request_session_http,
            base_url=base_url,
            ocp_user_token=ocp_token_for_actor,
            username="someotheruser",
        )
        assert bulk_resp.status_code == 403, (
            f"Expected 403 (non-admin cannot bulk revoke other users), "
            f"got {bulk_resp.status_code}: {bulk_resp.text[:200]}"
        )
        LOGGER.info("[bulk-revoke] Non-admin correctly received 403 when attempting to bulk revoke another user's keys")

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_bulk_revoke_admin_can_revoke_any_user(
        self,
        request_session_http: requests.Session,
        base_url: str,
        active_api_key_id: str,
        free_user_username: str,
        admin_ocp_token: str,
    ) -> None:
        """Verify an admin can bulk revoke any user's active API keys."""
        revoked_count = assert_bulk_revoke_success(
            request_session_http=request_session_http,
            base_url=base_url,
            ocp_user_token=admin_ocp_token,
            username=free_user_username,
            min_revoked_count=1,
        )
        LOGGER.info(f"[bulk-revoke] Admin successfully revoked {revoked_count} key(s) for user {free_user_username}")

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_bulk_revoke_by_subscription_only(
        self,
        request_session_http: requests.Session,
        base_url: str,
        admin_ocp_token: str,
        free_user_keys_across_subscriptions: FreeUserKeysAcrossSubscriptions,
    ) -> None:
        """Verify admin bulk revoke scoped to a subscription revokes only keys in that subscription."""
        key_setup = free_user_keys_across_subscriptions
        revoked_count = assert_bulk_revoke_by_subscription_success(
            request_session_http=request_session_http,
            base_url=base_url,
            ocp_user_token=admin_ocp_token,
            subscription=key_setup["primary_subscription_name"],
            min_revoked_count=len(key_setup["primary_subscription_key_ids"]),
        )
        LOGGER.info(
            f"[bulk-revoke] Admin revoked {revoked_count} key(s) for subscription "
            f"{key_setup['primary_subscription_name']}"
        )
        assert_api_keys_status(
            request_session_http=request_session_http,
            base_url=base_url,
            key_ids=key_setup["primary_subscription_key_ids"],
            ocp_user_token=admin_ocp_token,
            expected_status="revoked",
        )
        assert_api_keys_status(
            request_session_http=request_session_http,
            base_url=base_url,
            key_ids=key_setup["secondary_subscription_key_ids"],
            ocp_user_token=admin_ocp_token,
            expected_status="active",
        )

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_bulk_revoke_by_username_and_subscription(
        self,
        request_session_http: requests.Session,
        base_url: str,
        admin_ocp_token: str,
        free_user_keys_across_subscriptions: FreeUserKeysAcrossSubscriptions,
    ) -> None:
        """Verify username+subscription bulk revoke only affects that user's keys in that subscription."""
        key_setup = free_user_keys_across_subscriptions
        revoked_count = assert_bulk_revoke_success(
            request_session_http=request_session_http,
            base_url=base_url,
            ocp_user_token=admin_ocp_token,
            username=key_setup["username"],
            subscription=key_setup["primary_subscription_name"],
            min_revoked_count=len(key_setup["primary_subscription_key_ids"]),
        )
        LOGGER.info(
            f"[bulk-revoke] Admin revoked {revoked_count} key(s) for user {key_setup['username']} "
            f"in subscription {key_setup['primary_subscription_name']}"
        )
        assert_api_keys_status(
            request_session_http=request_session_http,
            base_url=base_url,
            key_ids=key_setup["primary_subscription_key_ids"],
            ocp_user_token=admin_ocp_token,
            expected_status="revoked",
        )
        assert_api_keys_status(
            request_session_http=request_session_http,
            base_url=base_url,
            key_ids=key_setup["secondary_subscription_key_ids"],
            ocp_user_token=admin_ocp_token,
            expected_status="active",
        )

    @pytest.mark.tier2
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_bulk_revoke_dry_run_by_username(
        self,
        request_session_http: requests.Session,
        base_url: str,
        admin_ocp_token: str,
        free_user_keys_across_subscriptions: FreeUserKeysAcrossSubscriptions,
    ) -> None:
        """Verify bulk-revoke dry-run by username previews matching keys without revoking them."""
        key_setup = free_user_keys_across_subscriptions
        all_key_ids = key_setup["primary_subscription_key_ids"] + key_setup["secondary_subscription_key_ids"]
        revoked_count = assert_bulk_dry_run_preview(
            request_session_http=request_session_http,
            base_url=base_url,
            ocp_user_token=admin_ocp_token,
            username=key_setup["username"],
            expected_revoked_count=len(all_key_ids),
        )
        LOGGER.info(f"[bulk-revoke] Dry-run by username returned revokedCount={revoked_count}")
        assert_api_keys_status(
            request_session_http=request_session_http,
            base_url=base_url,
            key_ids=all_key_ids,
            ocp_user_token=admin_ocp_token,
            expected_status="active",
        )

    @pytest.mark.tier2
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_bulk_revoke_dry_run_by_subscription(
        self,
        request_session_http: requests.Session,
        base_url: str,
        admin_ocp_token: str,
        free_user_keys_across_subscriptions: FreeUserKeysAcrossSubscriptions,
    ) -> None:
        """Verify bulk-revoke dry-run by subscription previews matching keys without revoking them."""
        key_setup = free_user_keys_across_subscriptions
        revoked_count = assert_bulk_dry_run_preview(
            request_session_http=request_session_http,
            base_url=base_url,
            ocp_user_token=admin_ocp_token,
            subscription=key_setup["primary_subscription_name"],
            expected_revoked_count=len(key_setup["primary_subscription_key_ids"]),
        )
        LOGGER.info(
            f"[bulk-revoke] Dry-run by subscription {key_setup['primary_subscription_name']} "
            f"returned revokedCount={revoked_count}"
        )
        assert_api_keys_status(
            request_session_http=request_session_http,
            base_url=base_url,
            key_ids=key_setup["primary_subscription_key_ids"] + key_setup["secondary_subscription_key_ids"],
            ocp_user_token=admin_ocp_token,
            expected_status="active",
        )
