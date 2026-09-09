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

    def test_scenario_1_spoofed_headers_with_valid_token(self, http_client):
        """Scenario 1: Valid auth token + spoofed headers through Gateway.

        Security Property: Authorino's AuthPolicy MUST overwrite client-supplied
        `x-user-id` and `x-tenant-id` headers with authenticated token review identity.
        """
        headers = {
            "Authorization": f"Bearer {VALID_SA_TOKEN}",
            "x-user-id": SPOOFED_USER,
            "x-tenant-id": SPOOFED_TENANT,
            "Content-Type": "application/json",
        }

        # Send request to Gateway endpoint
        target_url = f"{GATEWAY_URL.rstrip('/')}/v1/health"
        response = http_client.get(target_url, headers=headers)

        assert response.status_code == 200, (
            f"Expected 200 OK for valid SA token through Gateway, got {response.status_code}. "
            f"Response body: {response.text}"
        )

        # If endpoint returns echo or debug headers/context, verify overwritten identity
        resp_headers = {k.lower(): v for k, v in response.headers.items()}
        if "x-user-id" in resp_headers:
            assert resp_headers["x-user-id"] != SPOOFED_USER, (
                f"SECURITY VIOLATION: Spoofed x-user-id '{SPOOFED_USER}' was passed through! "
                f"Authorino AuthPolicy failed to overwrite identity header."
            )
            assert resp_headers["x-user-id"] == EXPECTED_USER, (
                f"Expected x-user-id '{EXPECTED_USER}', got '{resp_headers['x-user-id']}'"
            )

        if "x-tenant-id" in resp_headers:
            assert resp_headers["x-tenant-id"] != SPOOFED_TENANT, (
                f"SECURITY VIOLATION: Spoofed x-tenant-id '{SPOOFED_TENANT}' was passed through! "
                f"Authorino AuthPolicy failed to overwrite tenant header."
            )
            assert resp_headers["x-tenant-id"] == EXPECTED_TENANT, (
                f"Expected x-tenant-id '{EXPECTED_TENANT}', got '{resp_headers['x-tenant-id']}'"
            )

    def test_scenario_2_spoofed_headers_without_token(self, http_client):
        """Scenario 2: Spoofed headers with NO auth token through Gateway.

        Security Property: Gateway Authorino AuthPolicy MUST reject unauthenticated
        requests requiring authentication, even if client supplies spoofed identity headers.
        """
        headers = {
            "x-user-id": SPOOFED_USER,
            "x-tenant-id": SPOOFED_TENANT,
            "Content-Type": "application/json",
        }

        target_url = f"{GATEWAY_URL.rstrip('/')}/v1/health"
        response = http_client.get(target_url, headers=headers)

        assert response.status_code == 401, (
            f"SECURITY VIOLATION: Expected 401 Unauthorized for request with no token, "
            f"got {response.status_code}. Gateway allowed unauthenticated request with spoofed headers!"
        )

    def test_scenario_3_direct_access_bypassing_gateway(self, http_client):
        """Scenario 3: Direct request to OGX bypassing the Gateway.

        Security Property: OGX upstream_header auth provider MUST reject requests
        originating from non-trusted CIDRs (trusted_proxy_cidrs enforcement),
        preventing attackers from bypassing gateway auth filters.
        """
        headers = {
            "x-user-id": SPOOFED_USER,
            "x-tenant-id": SPOOFED_TENANT,
            "Content-Type": "application/json",
        }

        target_url = f"{DIRECT_OGX_URL.rstrip('/')}/v1/health"
        response = http_client.get(target_url, headers=headers)

        assert response.status_code == 403, (
            f"SECURITY VIOLATION: Expected 403 Forbidden for direct request bypassing Gateway, "
            f"got {response.status_code}. OGX trusted_proxy_cidrs failed to reject untrusted source IP!"
        )
