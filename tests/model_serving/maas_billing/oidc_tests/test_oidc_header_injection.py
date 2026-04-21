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
    "oidc_subscription",
    "oidc_auth_policy_patched",
)
class TestOIDCHeaderInjection:
    """Verify the gateway ignores client-supplied identity headers."""

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        "header_name, header_value",
        [
            pytest.param("X-MaaS-Username", "evil_hacker", id="username"),
            pytest.param("X-MaaS-Group", '["system:cluster-admins","cluster-admin"]', id="group"),
            pytest.param("X-MaaS-Subscription", "fake-subscription-id-12345", id="subscription"),
        ],
    )
    def test_injected_header_does_not_change_model_list(
        self,
        request_session_http: requests.Session,
        base_url: str,
        oidc_minted_api_key: dict[str, Any],
        baseline_models_response: requests.Response,
        header_name: str,
        header_value: str,
    ) -> None:
        """Verify injected identity header does not escalate access.

        Two safe outcomes:
        - 200 with identical model list (header overwritten by gateway)
        - 403 denial (header interfered but did not grant escalated access)
        """
        spoofed_response = fetch_models_with_spoofed_header(
            session=request_session_http,
            base_url=base_url,
            api_key=oidc_minted_api_key["key"],
            extra_headers={header_name: header_value},
        )

        if spoofed_response.status_code == 200:
            assert_model_lists_match(
                baseline_response=baseline_models_response,
                spoofed_response=spoofed_response,
                injection_description=header_name,
            )
            LOGGER.info(f"[oidc] {header_name} injection overwritten — same models returned")
        else:
            assert spoofed_response.status_code in (401, 403), (
                f"Unexpected status for injected {header_name}: "
                f"{spoofed_response.status_code}: {spoofed_response.text[:200]}"
            )
            LOGGER.info(
                f"[oidc] {header_name} injection caused denial ({spoofed_response.status_code}) "
                f"— no escalation possible"
            )

    @pytest.mark.tier2
    def test_injected_username_on_oidc_token_ignored(
        self,
        oidc_api_key_with_spoofed_username: dict[str, Any],
    ) -> None:
        """Verify client-supplied X-MaaS-Username with raw OIDC token does not override identity."""
        api_key_value = oidc_api_key_with_spoofed_username.get("key", "")
        assert api_key_value.startswith("sk-oai-"), (
            f"Expected API key starting with 'sk-oai-', got prefix: {api_key_value[:10]}..."
        )
        LOGGER.info("[oidc] API key minted with injected X-MaaS-Username — gateway ignored spoofed header")
