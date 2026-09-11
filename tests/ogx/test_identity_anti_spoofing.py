"""Integration test suite for RHAIENG-6746: Identity Header Anti-Spoofing.

Verifies end-to-end anti-spoofing behavior across Gateway (Authorino AuthPolicy)
and OGX backend (upstream_header auth provider with trusted_proxy_cidrs).

Scenarios tested:
1. Scenario 1: Spoofed identity headers with valid auth token -> Gateway overwrites spoofed headers (200 OK).
2. Scenario 2: Spoofed identity headers with NO auth token -> Gateway rejects request (401 Unauthorized).
3. Scenario 3: Direct access bypassing gateway -> OGX rejects request via trusted_proxy_cidrs (403 Forbidden).
"""

import os

import httpx
import pytest

# Configuration via environment variables with defaults for test harnesses
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8080")
DIRECT_OGX_URL = os.getenv("DIRECT_OGX_URL", "http://localhost:8000")
VALID_SA_TOKEN = os.getenv("K8S_SA_TOKEN", "valid-test-token-alice")
EXPECTED_USER = os.getenv("EXPECTED_USER", "system:serviceaccount:acme:alice")
EXPECTED_TENANT = os.getenv("EXPECTED_TENANT", "acme")

SPOOFED_USER = "evil-admin"
SPOOFED_TENANT = "other-tenant"


@pytest.fixture
def http_client():
    """HTTP client fixture for testing."""
    with httpx.Client(timeout=10.0, follow_redirects=False) as client:
        yield client


class TestIdentityHeaderAntiSpoofing:
    """Test suite for validating identity header anti-spoofing guarantees."""

    @pytest.mark.parametrize(
        "scenario_name, base_url, token, expected_status, check_identity_headers",
        [
            (
                "Scenario 1: Valid auth token + spoofed headers through Gateway",
                GATEWAY_URL,
                VALID_SA_TOKEN,
                200,
                True,
            ),
            (
                "Scenario 2: Spoofed headers with NO auth token through Gateway",
                GATEWAY_URL,
                None,
                401,
                False,
            ),
            (
                "Scenario 3: Direct request to OGX bypassing Gateway",
                DIRECT_OGX_URL,
                None,
                403,
                False,
            ),
        ],
        ids=[
            "valid_token_gateway_overwrites_headers",
            "missing_token_gateway_returns_401",
            "direct_access_ogx_returns_403",
        ],
    )
    def test_identity_header_anti_spoofing(
        self,
        http_client,
        scenario_name,
        base_url,
        token,
        expected_status,
        check_identity_headers,
    ):
        """Verify identity header anti-spoofing behavior across Gateway and OGX backend."""
        headers = {
            "x-user-id": SPOOFED_USER,
            "x-tenant-id": SPOOFED_TENANT,
            "Content-Type": "application/json",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        target_url = f"{base_url.rstrip('/')}/v1/health"
        response = http_client.get(target_url, headers=headers)

        assert response.status_code == expected_status, (
            f"[{scenario_name}] Expected {expected_status}, got {response.status_code}. Response body: {response.text}"
        )

        if check_identity_headers:
            response_headers = {
                header_key.lower(): header_value for header_key, header_value in response.headers.items()
            }
            if "x-user-id" in response_headers:
                assert response_headers["x-user-id"] != SPOOFED_USER, (
                    f"SECURITY VIOLATION: Spoofed x-user-id '{SPOOFED_USER}' was passed through! "
                    "Authorino AuthPolicy failed to overwrite identity header."
                )
                assert response_headers["x-user-id"] == EXPECTED_USER, (
                    f"Expected x-user-id '{EXPECTED_USER}', got '{response_headers['x-user-id']}'"
                )

            if "x-tenant-id" in response_headers:
                assert response_headers["x-tenant-id"] != SPOOFED_TENANT, (
                    f"SECURITY VIOLATION: Spoofed x-tenant-id '{SPOOFED_TENANT}' was passed through! "
                    "Authorino AuthPolicy failed to overwrite tenant header."
                )
                assert response_headers["x-tenant-id"] == EXPECTED_TENANT, (
                    f"Expected x-tenant-id '{EXPECTED_TENANT}', got '{response_headers['x-tenant-id']}'"
                )
