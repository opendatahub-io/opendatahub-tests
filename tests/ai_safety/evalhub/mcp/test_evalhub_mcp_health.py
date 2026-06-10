import pytest
from ocp_resources.route import Route

from tests.ai_safety.evalhub.mcp.constants import (
    EVALHUB_MCP_HEALTH_PATH,
    EVALHUB_MCP_HEALTH_STATUS_OK,
)
from tests.ai_safety.evalhub.mcp.utils import EvalHubMcpClient


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-mcp-health"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.ai_safety
@pytest.mark.usefixtures("evalhub_mcp_mt_ready")
class TestEvalHubMcpHealth:
    """Tests for the unauthenticated EvalHub MCP health endpoint."""

    def test_mcp_health_returns_ok(
        self,
        evalhub_mcp_mt_route: Route,
        evalhub_mcp_mt_ca_bundle_file: str,
    ) -> None:
        """Given the MCP route is reachable, /health returns status ok without authentication."""
        client = EvalHubMcpClient(
            host=evalhub_mcp_mt_route.host,
            token="unused",
            ca_bundle_file=evalhub_mcp_mt_ca_bundle_file,
            tenant="unused",
        )
        response = client.get_health(path=EVALHUB_MCP_HEALTH_PATH)
        response.raise_for_status()
        assert response.json()["status"] == EVALHUB_MCP_HEALTH_STATUS_OK

    def test_mcp_health_rejects_post(
        self,
        evalhub_mcp_mt_route: Route,
        evalhub_mcp_mt_ca_bundle_file: str,
    ) -> None:
        """Given the MCP health endpoint, POST requests are not served as healthy checks."""
        client = EvalHubMcpClient(
            host=evalhub_mcp_mt_route.host,
            token="unused",
            ca_bundle_file=evalhub_mcp_mt_ca_bundle_file,
            tenant="unused",
        )
        response = client.post_without_auth(
            method="initialize",
            params={},
            extra_headers={"Content-Type": "application/json"},
        )
        # POST to / goes through MCP auth; health is GET-only on /health.
        assert response.status_code in (401, 403, 405), (
            f"Expected POST without auth to be rejected, got {response.status_code}"
        )
