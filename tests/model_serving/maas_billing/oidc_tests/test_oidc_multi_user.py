from __future__ import annotations

from typing import Any

import pytest
import requests
import structlog

from tests.model_serving.maas_billing.oidc_tests.utils import (
    decode_jwt_payload,
    get_active_key_ids,
)

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "oidc_auth_policy_patched",
)
class TestOIDCMultiUser:
    """Verify OIDC multi-user scenarios: JWT claim mapping, and key isolation."""

    @pytest.mark.tier1
    def test_jwt_contains_preferred_username(
        self,
        external_oidc_token: str,
        oidc_user_credentials: dict[str, str],
    ) -> None:
        """Verify JWT preferred_username claim matches the authenticated OIDC user."""
        jwt_payload = decode_jwt_payload(token=external_oidc_token)

        assert "preferred_username" in jwt_payload, (
            f"JWT missing 'preferred_username' claim. Available claims: {list(jwt_payload.keys())}"
        )

        expected_username = oidc_user_credentials["username"]
        actual_username = jwt_payload["preferred_username"]
        assert actual_username == expected_username, (
            f"Expected preferred_username='{expected_username}', got '{actual_username}'"
        )
        LOGGER.info(f"[oidc] JWT preferred_username='{actual_username}' matches expected user")

    @pytest.mark.tier2
    def test_jwt_contains_groups_claim(
        self,
        external_oidc_token: str,
    ) -> None:
        """Verify JWT contains a groups or realm_access claim for RBAC."""
        jwt_payload = decode_jwt_payload(token=external_oidc_token)

        has_groups = "groups" in jwt_payload
        has_realm_access = "realm_access" in jwt_payload and "roles" in jwt_payload.get("realm_access", {})

        assert has_groups or has_realm_access, (
            f"JWT missing group information. Expected 'groups' or 'realm_access.roles'. "
            f"Available claims: {list(jwt_payload.keys())}"
        )

        if has_groups:
            groups = jwt_payload["groups"]
            LOGGER.info(f"[oidc] JWT contains 'groups' claim: {groups}")
        else:
            roles = jwt_payload["realm_access"]["roles"]
            LOGGER.info(f"[oidc] JWT contains 'realm_access.roles': {roles}")

    @pytest.mark.tier2
    def test_both_users_have_group_claims(
        self,
        external_oidc_token: str,
        second_user_oidc_token: str,
    ) -> None:
        """Verify two OIDC users have distinct group memberships in their JWTs."""
        first_user_payload = decode_jwt_payload(token=external_oidc_token)
        second_user_payload = decode_jwt_payload(token=second_user_oidc_token)

        first_user_groups = set(first_user_payload.get("groups", []))
        second_user_groups = set(second_user_payload.get("groups", []))

        first_username = first_user_payload.get("preferred_username", "user1")
        second_username = second_user_payload.get("preferred_username", "user2")

        LOGGER.info(f"[oidc] {first_username} groups: {first_user_groups}")
        LOGGER.info(f"[oidc] {second_username} groups: {second_user_groups}")

        assert first_user_groups and second_user_groups, (
            f"Both users must have groups. {first_username}={first_user_groups}, {second_username}={second_user_groups}"
        )

    @pytest.mark.tier2
    @pytest.mark.usefixtures("oidc_subscription")
    def test_second_user_can_mint_api_key(
        self,
        second_user_minted_api_key: dict[str, Any],
        oidc_second_user_credentials: dict[str, str],
    ) -> None:
        """Verify a second OIDC user can independently mint an API key."""
        assert second_user_minted_api_key.get("key", "").startswith("sk-oai-"), (
            "Expected API key starting with 'sk-oai-' prefix"
        )
        LOGGER.info(
            f"[oidc] {oidc_second_user_credentials['username']} created API key id={second_user_minted_api_key['id']}"
        )

    @pytest.mark.tier2
    @pytest.mark.usefixtures("oidc_subscription")
    def test_api_key_isolation_between_users(
        self,
        request_session_http: requests.Session,
        base_url: str,
        external_oidc_token: str,
        second_user_oidc_token: str,
        oidc_minted_api_key: dict[str, Any],
        second_user_minted_api_key: dict[str, Any],
    ) -> None:
        """Verify each OIDC user only sees their own API keys, not the other user's."""
        first_user_key_id = oidc_minted_api_key["id"]
        second_user_key_id = second_user_minted_api_key["id"]

        first_user_key_ids = get_active_key_ids(
            request_session_http=request_session_http,
            base_url=base_url,
            ocp_user_token=external_oidc_token,
        )
        assert first_user_key_id in first_user_key_ids, (
            f"First user's key {first_user_key_id} not found in their own key list"
        )
        assert second_user_key_id not in first_user_key_ids, (
            f"Second user's key {second_user_key_id} should NOT be visible to first user"
        )

        second_user_key_ids = get_active_key_ids(
            request_session_http=request_session_http,
            base_url=base_url,
            ocp_user_token=second_user_oidc_token,
        )
        assert second_user_key_id in second_user_key_ids, (
            f"Second user's key {second_user_key_id} not found in their own key list"
        )
        assert first_user_key_id not in second_user_key_ids, (
            f"First user's key {first_user_key_id} should NOT be visible to second user"
        )

        LOGGER.info(
            f"[oidc] Key isolation verified: user1 sees {len(first_user_key_ids)} keys, "
            f"user2 sees {len(second_user_key_ids)} keys, no cross-visibility"
        )
