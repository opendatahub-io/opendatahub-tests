import json

import pytest
import requests
from ocp_resources.deployment import Deployment

from tests.rhoai_mcp.constants import RHOAI_MCP_HEALTH_PATH


class TestRhoaiMcpDeployment:
    """Verify rhoai-mcp deploys successfully and becomes healthy."""

    @pytest.mark.smoke
    def test_deployment_replicas_ready(self, rhoai_mcp_deployment: Deployment) -> None:
        """Given rhoai-mcp resources are applied to the cluster
        When the Deployment readiness is checked
        Then all replicas report ready
        """
        assert rhoai_mcp_deployment.exists

    @pytest.mark.smoke
    def test_health_endpoint(
        self,
        rhoai_mcp_base_url: str,
        rhoai_mcp_ca_bundle: str,
        rhoai_mcp_ready: None,
    ) -> None:
        """Given rhoai-mcp is deployed and running
        When the /health endpoint is polled via the Route
        Then it returns a healthy response
        """
        url = f"{rhoai_mcp_base_url}{RHOAI_MCP_HEALTH_PATH}"
        response = requests.get(url, verify=rhoai_mcp_ca_bundle, timeout=10)
        assert response.ok

    @pytest.mark.smoke
    def test_unauthenticated_mcp_rejected(
        self,
        rhoai_mcp_endpoint_url: str,
        rhoai_mcp_ca_bundle: str,
        rhoai_mcp_ready: None,
    ) -> None:
        """Given rhoai-mcp is deployed with OIDC authentication enabled
        When an unauthenticated request is sent to the /mcp endpoint
        Then the server returns 401 with a WWW-Authenticate: Bearer header
        """
        response = requests.post(
            url=rhoai_mcp_endpoint_url,
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            verify=rhoai_mcp_ca_bundle,
            timeout=10,
        )
        assert response.status_code == 401
        assert "Bearer" in response.headers.get("WWW-Authenticate", "")

    @pytest.mark.smoke
    def test_authenticated_mcp_succeeds(
        self,
        rhoai_mcp_endpoint_url: str,
        rhoai_mcp_ca_bundle: str,
        rhoai_mcp_ready: None,
        current_client_token: str,
    ) -> None:
        """Given rhoai-mcp is deployed with OIDC authentication enabled
        When an authenticated JSON-RPC initialize request is sent to the /mcp endpoint
        Then the server returns 200 with server capabilities and a session ID
        """
        response = requests.post(
            url=rhoai_mcp_endpoint_url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "0.1.0"},
                },
            },
            headers={
                "Authorization": f"Bearer {current_client_token}",
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            verify=rhoai_mcp_ca_bundle,
            timeout=30,
        )
        assert response.status_code == 200
        data_lines = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
        assert data_lines, "No SSE data lines in response"
        body = json.loads(data_lines[0])
        assert body["jsonrpc"] == "2.0"
        assert body["id"] == 1
        result = body["result"]
        assert "serverInfo" in result
        assert "capabilities" in result
        assert response.headers.get("mcp-session-id")
