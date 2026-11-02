from typing import Any

import pytest
import requests

from tests.ai_gateway.models_as_a_service.multitenancy.utils import (
    assert_tenant_gateway_inference_auth_denied,
    assert_tenant_inference_status,
)

_FORGED_IDENTITY_HEADER_CASES: list[tuple[str, str]] = [
    ("X-MaaS-Username", "cluster-admin"),
    ("X-MaaS-Group", "system:cluster-admins,system:masters"),
    ("X-MaaS-KeyName", "forged-prod-key"),
    ("X-MaaS-Identity", '{"source":"oidc","claims":{}}'),
    ("X-MaaS-Key-Id", "fake-key-id-00000"),
]

_INVALID_API_KEY = "sk-oai-e2e-invalid-not-a-real-key"  # pragma: allowlist secret

_TENANT_GATEWAY_CASES = [
    pytest.param("a", id="tenant_a"),
    pytest.param("b", id="tenant_b"),
]


def _tenant_inference_url(tenant_id: str, tenant_a_inference_url: str, tenant_b_inference_url: str) -> str:
    return tenant_a_inference_url if tenant_id == "a" else tenant_b_inference_url


def _tenant_inference_headers(
    tenant_id: str,
    tenant_a_inference_headers: dict[str, str],
    tenant_b_inference_headers: dict[str, str],
) -> dict[str, str]:
    return tenant_a_inference_headers if tenant_id == "a" else tenant_b_inference_headers


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
    @pytest.mark.parametrize("tenant_id", _TENANT_GATEWAY_CASES)
    def test_missing_api_key_denied_on_each_tenant_gateway(
        self,
        tenant_id: str,
        request_session_http: requests.Session,
        tenant_a_inference_url: str,
        tenant_b_inference_url: str,
        tenant_inference_chat_payload: dict[str, Any],
    ) -> None:
        """Given a governed tenant inference URL, when Authorization is omitted on that gateway,
        then the request returns 401 or 403.
        """
        inference_url = _tenant_inference_url(
            tenant_id=tenant_id,
            tenant_a_inference_url=tenant_a_inference_url,
            tenant_b_inference_url=tenant_b_inference_url,
        )
        headers = {"Content-Type": "application/json"}
        assert_tenant_gateway_inference_auth_denied(
            session=request_session_http,
            inference_url=inference_url,
            headers=headers,
            payload=tenant_inference_chat_payload,
            scenario_label="missing API key",
        )

    @pytest.mark.tier2
    @pytest.mark.parametrize("tenant_id", _TENANT_GATEWAY_CASES)
    def test_invalid_api_key_denied_on_each_tenant_gateway(
        self,
        tenant_id: str,
        request_session_http: requests.Session,
        tenant_a_inference_url: str,
        tenant_b_inference_url: str,
        tenant_inference_chat_payload: dict[str, Any],
    ) -> None:
        """Given a governed tenant inference URL, when an invalid sk-oai-* key is used on that gateway,
        then the request returns 401 or 403.
        """
        inference_url = _tenant_inference_url(
            tenant_id=tenant_id,
            tenant_a_inference_url=tenant_a_inference_url,
            tenant_b_inference_url=tenant_b_inference_url,
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_INVALID_API_KEY}",
        }
        assert_tenant_gateway_inference_auth_denied(
            session=request_session_http,
            inference_url=inference_url,
            headers=headers,
            payload=tenant_inference_chat_payload,
            scenario_label="invalid API key",
        )

    @pytest.mark.tier2
    @pytest.mark.parametrize("tenant_id", _TENANT_GATEWAY_CASES)
    @pytest.mark.parametrize(
        "header_name, header_value",
        [
            pytest.param(name, value, id=name.removeprefix("X-MaaS-").lower().replace("-", "_"))
            for name, value in _FORGED_IDENTITY_HEADER_CASES
        ],
    )
    def test_forged_identity_header_denied_on_tenant_gateway(
        self,
        tenant_id: str,
        header_name: str,
        header_value: str,
        request_session_http: requests.Session,
        tenant_a_inference_url: str,
        tenant_b_inference_url: str,
        tenant_a_inference_headers: dict[str, str],
        tenant_b_inference_headers: dict[str, str],
        tenant_inference_chat_payload: dict[str, Any],
    ) -> None:
        """Given a valid tenant API key, when a client-supplied X-MaaS identity header is sent on that gateway,
        then inference is denied with 401 or 403 (RHOAIENG-99369 class).
        """
        inference_url = _tenant_inference_url(
            tenant_id=tenant_id,
            tenant_a_inference_url=tenant_a_inference_url,
            tenant_b_inference_url=tenant_b_inference_url,
        )
        base_headers = _tenant_inference_headers(
            tenant_id=tenant_id,
            tenant_a_inference_headers=tenant_a_inference_headers,
            tenant_b_inference_headers=tenant_b_inference_headers,
        )
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
            header_name: header_value,
        }
        assert_tenant_gateway_inference_auth_denied(
            session=request_session_http,
            inference_url=inference_url,
            headers=headers,
            payload=tenant_inference_chat_payload,
            scenario_label=f"forged {header_name}",
        )
