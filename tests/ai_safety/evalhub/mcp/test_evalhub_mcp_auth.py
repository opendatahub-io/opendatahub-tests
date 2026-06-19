import pytest
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route

from tests.ai_safety.evalhub.mcp.utils import EvalHubMcpClient, validate_evalhub_mcp_initialize
from tests.ai_safety.evalhub.utils import TENANT_HEADER, build_headers
from utilities.guardrails import get_auth_headers


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-mcp-auth"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier2
@pytest.mark.ai_safety
@pytest.mark.usefixtures("evalhub_mcp_mt_ready", "evalhub_mcp_proxy_role_binding")
class TestEvalHubMcpAuth:
    """Authentication tests for evalhub-mcp behind kube-rbac-proxy."""

    def test_mcp_request_with_valid_token_succeeds(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mcp_mt_route: Route,
        evalhub_mcp_mt_ca_bundle_file: str,
    ) -> None:
        """
        Given: A valid bearer token, X-Tenant header, and evalhubs/proxy RBAC
        When: MCP initialize request is sent
        Then: Server accepts the request and returns initialize result
        """
        result = validate_evalhub_mcp_initialize(
            host=evalhub_mcp_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mcp_mt_ca_bundle_file,
            tenant_namespace=tenant_a_namespace.name,
        )
        assert result.get("serverInfo"), "Expected serverInfo in initialize result"

    def test_mcp_request_without_token_is_rejected(
        self,
        evalhub_mcp_mt_route: Route,
        evalhub_mcp_mt_ca_bundle_file: str,
        tenant_a_namespace: Namespace,
    ) -> None:
        """
        Given: No Authorization header is provided
        When: MCP initialize request is sent
        Then: Server rejects with HTTP 401
        """
        client = EvalHubMcpClient(
            host=evalhub_mcp_mt_route.host,
            token="unused",
            ca_bundle_file=evalhub_mcp_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
        )
        response = client.post_without_auth(
            method="initialize",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "auth-test", "version": "1.0"},
            },
        )
        assert response.status_code == 401, f"Expected 401 without token, got {response.status_code}: {response.text}"

    def test_mcp_request_without_tenant_header_is_rejected(
        self,
        tenant_a_token: str,
        evalhub_mcp_mt_route: Route,
        evalhub_mcp_mt_ca_bundle_file: str,
    ) -> None:
        """
        Given: A valid bearer token is provided without X-Tenant header
        When: MCP initialize request is sent
        Then: Server rejects with HTTP 403
        """
        headers = get_auth_headers(token=tenant_a_token)
        response = EvalHubMcpClient(
            host=evalhub_mcp_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mcp_mt_ca_bundle_file,
            tenant="unused",
        ).post_without_auth(
            method="initialize",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "auth-test", "version": "1.0"},
            },
            extra_headers=headers,
        )
        assert response.status_code == 403, (
            f"Expected 403 without X-Tenant, got {response.status_code}: {response.text}"
        )


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-mcp-auth-denied"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier2
@pytest.mark.ai_safety
@pytest.mark.usefixtures("evalhub_mcp_mt_ready")
class TestEvalHubMcpProxyRbac:
    """Authorization tests for evalhubs/proxy RBAC on the MCP route."""

    def test_mcp_request_without_proxy_rbac_is_rejected(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mcp_mt_route: Route,
        evalhub_mcp_mt_ca_bundle_file: str,
    ) -> None:
        """
        Given: A valid token and X-Tenant header without evalhubs/proxy RBAC
        When: MCP initialize request is sent
        Then: kube-rbac-proxy rejects with HTTP 403
        """
        headers = build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name)
        assert TENANT_HEADER in headers
        response = EvalHubMcpClient(
            host=evalhub_mcp_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mcp_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
        ).post_without_auth(
            method="initialize",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "auth-test", "version": "1.0"},
            },
            extra_headers=headers,
        )
        assert response.status_code == 403, (
            f"Expected 403 without proxy RBAC, got {response.status_code}: {response.text}"
        )
