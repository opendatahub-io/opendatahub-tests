from typing import Any

import pytest
import requests

from tests.ai_gateway.models_as_a_service.multitenancy.utils import (
    assert_tenant_gateway_inference_auth_denied,
    assert_tenant_inference_status,
)

_FORGED_IDENTITY_HEADERS: dict[str, str] = {
    "X-MaaS-Username": "cluster-admin",
    "X-MaaS-Group": "system:cluster-admins,system:masters",
    "X-MaaS-KeyName": "forged-prod-key",
    "X-MaaS-Identity": '{"source":"oidc","claims":{}}',
    "X-MaaS-Key-Id": "fake-key-id-00000",
}

_INVALID_API_KEY = "sk-oai-e2e-invalid-not-a-real-key"  # pragma: allowlist secret


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "aitenant_infra_namespace",
    "isolation_tenant_governance",
    "per_tenant_maas_api_ready",
)
class TestMultitenancyTenantGatewayNegativeSecurity:
    """Negative auth on per-tenant gateway URLs (RHOAIENG-100758).

    Complements default-gateway negatives in maas_api_key/ and oidc_tests/ and
    cross-tenant denial in multitenancy/inference/test_tenant_model_inference.py.
    """

    @pytest.mark.tier2
    def test_missing_api_key_denied_on_each_tenant_gateway(
        self,
        request_session_http: requests.Session,
        tenant_a_inference_url: str,
        tenant_b_inference_url: str,
        tenant_inference_chat_payload: dict[str, Any],
    ) -> None:
        """Given governed tenant inference URLs, when Authorization is omitted on each gateway,
        then both return 401 or 403.
        """
        headers = {"Content-Type": "application/json"}
        for inference_url in (tenant_a_inference_url, tenant_b_inference_url):
            assert_tenant_gateway_inference_auth_denied(
                session=request_session_http,
                inference_url=inference_url,
                headers=headers,
                payload=tenant_inference_chat_payload,
                scenario_label="missing API key",
            )

    @pytest.mark.tier2
    def test_invalid_api_key_denied_on_each_tenant_gateway(
        self,
        request_session_http: requests.Session,
        tenant_a_inference_url: str,
        tenant_b_inference_url: str,
        tenant_inference_chat_payload: dict[str, Any],
    ) -> None:
        """Given governed tenant inference URLs, when an invalid sk-oai-* key is used on each gateway,
        then both return 401 or 403.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_INVALID_API_KEY}",
        }
        for inference_url in (tenant_a_inference_url, tenant_b_inference_url):
            assert_tenant_gateway_inference_auth_denied(
                session=request_session_http,
                inference_url=inference_url,
                headers=headers,
                payload=tenant_inference_chat_payload,
                scenario_label="invalid API key",
            )

    @pytest.mark.tier2
    def test_forged_identity_headers_denied_on_each_tenant_gateway(
        self,
        request_session_http: requests.Session,
        tenant_a_inference_url: str,
        tenant_b_inference_url: str,
        tenant_a_inference_headers: dict[str, str],
        tenant_b_inference_headers: dict[str, str],
        tenant_inference_chat_payload: dict[str, Any],
    ) -> None:
        """Given valid tenant API keys, when client-supplied X-MaaS identity headers are sent on each gateway,
        then inference is denied with 401 or 403 (RHOAIENG-99369 class).
        """
        cases = (
            (tenant_a_inference_url, tenant_a_inference_headers),
            (tenant_b_inference_url, tenant_b_inference_headers),
        )
        for inference_url, base_headers in cases:
            control_headers = {**base_headers, "Content-Type": "application/json"}
            assert_tenant_inference_status(
                session=request_session_http,
                inference_url=inference_url,
                headers=control_headers,
                payload=tenant_inference_chat_payload,
                expected_status=200,
            )
            headers = {
                **control_headers,
                **_FORGED_IDENTITY_HEADERS,
            }
            assert_tenant_gateway_inference_auth_denied(
                session=request_session_http,
                inference_url=inference_url,
                headers=headers,
                payload=tenant_inference_chat_payload,
                scenario_label="forged identity headers",
            )
