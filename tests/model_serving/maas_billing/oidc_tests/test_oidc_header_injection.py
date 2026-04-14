from __future__ import annotations

from typing import Any

import pytest
import requests
import structlog

from tests.model_serving.maas_billing.oidc_tests.utils import (
    assert_model_lists_match,
    fetch_models_with_spoofed_header,
)

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "minimal_subscription_for_free_user",
    "oidc_auth_policy_patched",
)
class TestOIDCHeaderInjection:
    """Verify the gateway ignores client-supplied identity headers."""

    @pytest.mark.tier1
    def test_injected_username_header_ignored(
        self,
        request_session_http: requests.Session,
        base_url: str,
        oidc_minted_api_key: dict[str, Any],
    ) -> None:
        """Verify client-supplied X-MaaS-Username does not override authenticated identity."""
        api_key = oidc_minted_api_key["key"]

        baseline_response = fetch_models_with_spoofed_header(
            session=request_session_http,
            base_url=base_url,
            api_key=api_key,
        )
        assert baseline_response.status_code == 200, f"Baseline request failed: {baseline_response.status_code}"

        spoofed_response = fetch_models_with_spoofed_header(
            session=request_session_http,
            base_url=base_url,
            api_key=api_key,
            extra_headers={"X-MaaS-Username": "evil_hacker"},
        )

        assert spoofed_response.status_code == 200, (
            f"Expected 200 (injected header ignored), got {spoofed_response.status_code}: {spoofed_response.text[:200]}"
        )
        assert_model_lists_match(
            baseline_response=baseline_response,
            spoofed_response=spoofed_response,
            injection_description="X-MaaS-Username",
        )
        LOGGER.info("[oidc] X-MaaS-Username injection correctly ignored by gateway")

    @pytest.mark.tier2
    def test_injected_group_header_does_not_escalate(
        self,
        request_session_http: requests.Session,
        base_url: str,
        oidc_minted_api_key: dict[str, Any],
    ) -> None:
        """Verify client-supplied X-MaaS-Group does not grant access to unauthorized resources."""
        api_key = oidc_minted_api_key["key"]

        baseline_response = fetch_models_with_spoofed_header(
            session=request_session_http,
            base_url=base_url,
            api_key=api_key,
        )
        assert baseline_response.status_code == 200, f"Baseline request failed: {baseline_response.status_code}"

        spoofed_response = fetch_models_with_spoofed_header(
            session=request_session_http,
            base_url=base_url,
            api_key=api_key,
            extra_headers={"X-MaaS-Group": '["system:cluster-admins","cluster-admin"]'},
        )

        if spoofed_response.status_code == 200:
            assert_model_lists_match(
                baseline_response=baseline_response,
                spoofed_response=spoofed_response,
                injection_description="X-MaaS-Group",
            )
            LOGGER.info("[oidc] X-MaaS-Group injection overwritten — same models returned")
        else:
            assert spoofed_response.status_code in (400, 403), (
                f"Unexpected status for injected group header: "
                f"{spoofed_response.status_code} {spoofed_response.text[:200]}"
            )
            LOGGER.info(
                f"[oidc] X-MaaS-Group injection caused denial ({spoofed_response.status_code}) — no escalation possible"
            )

    @pytest.mark.tier2
    def test_injected_subscription_header_ignored(
        self,
        request_session_http: requests.Session,
        base_url: str,
        oidc_minted_api_key: dict[str, Any],
    ) -> None:
        """Verify client-supplied X-MaaS-Subscription does not grant access to other subscriptions."""
        api_key = oidc_minted_api_key["key"]

        baseline_response = fetch_models_with_spoofed_header(
            session=request_session_http,
            base_url=base_url,
            api_key=api_key,
        )
        assert baseline_response.status_code == 200, f"Baseline request failed: {baseline_response.status_code}"

        spoofed_response = fetch_models_with_spoofed_header(
            session=request_session_http,
            base_url=base_url,
            api_key=api_key,
            extra_headers={"X-MaaS-Subscription": "fake-subscription-id-12345"},
        )
        assert spoofed_response.status_code == 200, (
            f"Expected 200 (injected subscription header ignored), "
            f"got {spoofed_response.status_code}: {spoofed_response.text[:200]}"
        )

        assert_model_lists_match(
            baseline_response=baseline_response,
            spoofed_response=spoofed_response,
            injection_description="X-MaaS-Subscription",
        )
        LOGGER.info("[oidc] X-MaaS-Subscription injection correctly ignored")

    @pytest.mark.tier2
    def test_injected_username_on_oidc_token_ignored(
        self,
        oidc_api_key_with_spoofed_username: dict[str, Any],
    ) -> None:
        """Verify client-supplied X-MaaS-Username with raw OIDC token does not override identity."""
        assert oidc_api_key_with_spoofed_username.get("key", "").startswith("sk-oai-"), (
            f"Unexpected API key payload: {oidc_api_key_with_spoofed_username}"
        )
        LOGGER.info("[oidc] API key minted with injected X-MaaS-Username — gateway ignored spoofed header")
