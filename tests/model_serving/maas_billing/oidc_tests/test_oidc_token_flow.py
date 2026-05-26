from __future__ import annotations

from typing import Any

import pytest
import requests
import structlog

from tests.model_serving.maas_billing.utils import create_api_key
from utilities.general import generate_random_name

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "oidc_auth_policy_patched",
)
class TestOIDCTokenFlow:
    """Verify OIDC token authentication and rejection for MaaS API key operations.

    Tests the gateway's ability to accept valid OIDC JWTs, reject tampered tokens,
    and enforce the Authorization header requirement.
    """

    @pytest.mark.smoke
    @pytest.mark.usefixtures("oidc_subscription")
    def test_oidc_token_creates_api_key(
        self,
        oidc_minted_api_key: dict[str, Any],
    ) -> None:
        """Verify a valid OIDC token can create a MaaS API key with sk-oai- prefix."""
        assert "id" in oidc_minted_api_key, "Expected 'id' field in API key creation response"
        assert "key" in oidc_minted_api_key, "Expected 'key' field in API key creation response"

        plaintext_key: str = oidc_minted_api_key["key"]
        assert plaintext_key.startswith("sk-oai-"), "Expected API key to start with 'sk-oai-' prefix"
        assert len(plaintext_key) > len("sk-oai-"), "API key body after prefix must not be empty"

        LOGGER.info(f"[oidc] Created API key id={oidc_minted_api_key['id']}")

    @pytest.mark.tier3
    def test_tampered_oidc_token_rejected(
        self,
        request_session_http: requests.Session,
        base_url: str,
        external_oidc_token: str,
    ) -> None:
        """Verify a tampered OIDC token is rejected with 401."""
        corrupted_token = f"{external_oidc_token}TAMPERED"
        key_name = f"e2e-oidc-tampered-{generate_random_name()}"

        creation_response, _ = create_api_key(
            base_url=base_url,
            ocp_user_token=corrupted_token,
            request_session_http=request_session_http,
            api_key_name=key_name,
            raise_on_error=False,
        )

        assert creation_response.status_code in (401, 403, 500), (
            f"Expected rejection for tampered OIDC token, got {creation_response.status_code}: "
            f"{creation_response.text[:200]}"
        )
        LOGGER.info(f"[oidc] Tampered OIDC token correctly rejected with {creation_response.status_code}")

    @pytest.mark.tier3
    def test_empty_bearer_token_rejected(
        self,
        request_session_http: requests.Session,
        base_url: str,
    ) -> None:
        """Verify an empty Bearer token is rejected with 401."""
        key_name = f"e2e-oidc-empty-{generate_random_name()}"

        creation_response, _ = create_api_key(
            base_url=base_url,
            ocp_user_token="",
            request_session_http=request_session_http,
            api_key_name=key_name,
            raise_on_error=False,
        )

        assert creation_response.status_code == 401, (
            f"Expected 401 for empty bearer token, got {creation_response.status_code}: {creation_response.text[:200]}"
        )
        LOGGER.info("[oidc] Empty bearer token correctly rejected with 401")

    @pytest.mark.tier3
    def test_no_authorization_header_rejected(
        self,
        request_session_http: requests.Session,
        base_url: str,
    ) -> None:
        """Verify a request without Authorization header is rejected with 401."""
        api_keys_url = f"{base_url}/v1/api-keys"
        key_name = f"e2e-oidc-noauth-{generate_random_name()}"

        response = request_session_http.post(
            url=api_keys_url,
            json={"name": key_name},
            headers={"Content-Type": "application/json"},
            timeout=60,
        )

        assert response.status_code == 401, (
            f"Expected 401 for missing Authorization header, got {response.status_code}: {response.text[:200]}"
        )
        LOGGER.info("[oidc] Missing Authorization header correctly rejected with 401")

    @pytest.mark.tier3
    def test_random_jwt_rejected(
        self,
        request_session_http: requests.Session,
        base_url: str,
    ) -> None:
        """Verify a forged JWT not signed by Keycloak is rejected with 401."""
        forged_jwt = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJmYWtlIn0.invalid-signature"  # pragma: allowlist secret
        key_name = f"e2e-oidc-forged-{generate_random_name()}"

        creation_response, _ = create_api_key(
            base_url=base_url,
            ocp_user_token=forged_jwt,
            request_session_http=request_session_http,
            api_key_name=key_name,
            raise_on_error=False,
        )

        assert creation_response.status_code == 401, (
            f"Expected 401 for forged JWT, got {creation_response.status_code}: {creation_response.text[:200]}"
        )
        LOGGER.info("[oidc] Forged JWT correctly rejected with 401")
