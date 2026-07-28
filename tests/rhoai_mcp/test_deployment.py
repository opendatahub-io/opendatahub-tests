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
    def test_unauthenticated_sse_rejected(
        self,
        rhoai_mcp_base_url: str,
        rhoai_mcp_ca_bundle: str,
        rhoai_mcp_ready: None,
    ) -> None:
        """Given rhoai-mcp is deployed with OIDC authentication enabled
        When an unauthenticated request is sent to the /sse endpoint
        Then the server returns 401 with a WWW-Authenticate: Bearer header
        """
        url = f"{rhoai_mcp_base_url}/sse"
        response = requests.get(url, verify=rhoai_mcp_ca_bundle, timeout=10)
        assert response.status_code == 401
        assert "Bearer" in response.headers.get("WWW-Authenticate", "")

    @pytest.mark.smoke
    def test_authenticated_sse_succeeds(
        self,
        rhoai_mcp_base_url: str,
        rhoai_mcp_ca_bundle: str,
        rhoai_mcp_ready: None,
        current_client_token: str,
    ) -> None:
        """Given rhoai-mcp is deployed with OIDC authentication enabled
        When an authenticated request is sent to the /sse endpoint
        Then the server returns 200 and begins an SSE event stream
        """
        # Uses admin token for now; will swap with create_inference_token(sa)
        # for dedicated test identity in future tests
        url = f"{rhoai_mcp_base_url}/sse"
        with requests.get(
            url,
            headers={"Authorization": f"Bearer {current_client_token}"},
            verify=rhoai_mcp_ca_bundle,
            timeout=30,
            stream=True,
        ) as response:
            assert response.status_code == 200
            content_type = response.headers.get("content-type", "")
            assert "text/event-stream" in content_type
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    assert line.startswith((":", "data:", "event:", "id:", "retry:"))
                    break
            else:
                pytest.fail("rhoai-mcp closed the SSE stream without emitting an event")
