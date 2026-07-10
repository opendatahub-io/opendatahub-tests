from typing import Any

import pytest
import requests
from kubernetes.dynamic import DynamicClient

from tests.model_serving.maas_billing.multitenancy.aitenant.utils import AITenantTestContext
from tests.model_serving.maas_billing.multitenancy.utils import (
    TenantIsolationGovernance,
    assert_tenant_cross_tenant_inference_rejected,
    assert_tenant_inference_status,
    verify_tenant_model_httproutes_for_contexts,
)


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "aitenant_infra_namespace",
    "isolation_tenant_governance",
    "per_tenant_maas_api_ready",
)
class TestMultitenancyTenantModelInference:
    """Verify tenant-local model inference is routed and auth-scoped per tenant Gateway."""

    @pytest.mark.tier2
    def test_tenant_model_kserve_httproute_accepted_on_gateway(
        self,
        admin_client: DynamicClient,
        two_aitenant_test_contexts: tuple[AITenantTestContext, AITenantTestContext],
        isolation_tenant_governance: list[TenantIsolationGovernance],
    ) -> None:
        """Given tenant-local models provisioned per AITenant, when checking KServe HTTPRoutes,
        then each route is Accepted by its tenant Gateway.
        """
        verify_tenant_model_httproutes_for_contexts(
            admin_client=admin_client,
            test_contexts=two_aitenant_test_contexts,
            governance_contexts=isolation_tenant_governance,
        )

    @pytest.mark.tier2
    def test_inference_via_tenant_a_gateway_returns_200(
        self,
        request_session_http: requests.Session,
        tenant_a_inference_url: str,
        tenant_a_inference_headers: dict[str, str],
        tenant_inference_chat_payload: dict[str, Any],
    ) -> None:
        """Given a tenant-local model and API key in tenant A, when posting chat completions
        through tenant A's Gateway, then inference returns 200 OK.
        """
        assert_tenant_inference_status(
            session=request_session_http,
            inference_url=tenant_a_inference_url,
            headers=tenant_a_inference_headers,
            payload=tenant_inference_chat_payload,
            expected_status=200,
        )

    @pytest.mark.tier2
    def test_tenant_a_key_on_tenant_b_gateway_rejected(
        self,
        request_session_http: requests.Session,
        tenant_b_inference_url: str,
        tenant_a_inference_headers: dict[str, str],
        tenant_inference_chat_payload: dict[str, Any],
    ) -> None:
        """Given an API key minted in tenant A, when posting chat completions through tenant B's Gateway,
        then the gateway rejects the request with 401 or 403.
        """
        assert_tenant_cross_tenant_inference_rejected(
            session=request_session_http,
            inference_url=tenant_b_inference_url,
            headers=tenant_a_inference_headers,
            payload=tenant_inference_chat_payload,
        )
