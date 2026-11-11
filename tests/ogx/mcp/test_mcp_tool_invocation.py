"""MCP tool runtime: connector registration and server-side tool invocation.

Covers OGX's Model Context Protocol integration with an in-cluster MCP server:
startup registration as a connector (observable via the admin connectors API,
which performs a live MCP handshake), tool discovery, and the full responses-API
tool loop — a model tool-call executed by OGX against the MCP server, with the
tool result returned in the response.

The model is a mock that deterministically requests the mock MCP server's
``get_weather`` tool, so no external model or credentials are involved.
"""

import time

import httpx
import pytest
import structlog
from ogx_client import OgxClient

from tests.ogx.constants import OGX_CLIENT_VERIFY_SSL
from tests.ogx.mockupstream.constants import (
    MOCK_MAAS_QUALIFIED_MODEL_ID,
    MOCK_MCP_CONNECTOR_ID,
    MOCK_MCP_SERVER_LABEL,
    MOCK_MCP_TOOL_NAME,
    MOCK_MCP_TOOL_RESULT_TEXT,
)
from tests.ogx.mockupstream.deployments import MockUpstream

LOGGER = structlog.get_logger(name=__name__)

MCP_NAMESPACE = {"name": "test-ogx-mcp", "randomize_name": True}

# Model prompt that makes the mock MaaS request the get_weather tool (the mock
# extracts the city from "in <City>").
MCP_TOOL_PROMPT = "What is the weather in Boston?"


def _admin_get(ogx_admin_api_url: str, path: str) -> dict:
    """GET an admin API path, retrying transient failures (route warm-up)."""
    with httpx.Client(verify=OGX_CLIENT_VERIFY_SSL, timeout=120) as client:
        for attempt in range(3):
            response = client.get(f"{ogx_admin_api_url}{path}")
            if response.status_code < 500:
                break
            LOGGER.warning(
                "Transient server error from admin API, retrying",
                path=path,
                status=response.status_code,
                attempt=attempt + 1,
            )
            time.sleep(5)
        response.raise_for_status()
        return response.json()


@pytest.mark.ogx
@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [pytest.param(MCP_NAMESPACE)],
    indirect=True,
)
class TestMcpToolRuntime:
    """MCP connector registration, discovery and invocation."""

    @pytest.mark.tier2
    def test_mcp_connector_registered(self, ogx_admin_api_url: str) -> None:
        """Verify the MCP server is registered as a connector with live server info.

        Given: an OGX server whose run config registers the in-cluster mock
            MCP server as a connector.
        When: the admin connectors API is queried.
        Then: the connector is listed, and fetching it performs a live MCP
            initialize handshake (server name and version come from the MCP
            server itself).
        """
        connectors = _admin_get(ogx_admin_api_url=ogx_admin_api_url, path="/v1alpha/admin/connectors")
        connector_ids = [connector["connector_id"] for connector in connectors["data"]]
        assert MOCK_MCP_CONNECTOR_ID in connector_ids, f"Connector not registered: {connector_ids}"

        connector = _admin_get(
            ogx_admin_api_url=ogx_admin_api_url, path=f"/v1alpha/admin/connectors/{MOCK_MCP_CONNECTOR_ID}"
        )
        assert connector["server_name"] == "mock-mcp", f"MCP handshake failed: {connector}"
        assert connector["server_version"] == "1.0.0", f"Unexpected server version: {connector}"

    @pytest.mark.tier2
    def test_mcp_connector_tools_listed(self, ogx_admin_api_url: str) -> None:
        """Verify OGX discovers the MCP server's tools via a live tools/list call.

        Given: the registered MCP connector.
        When: the connector's tools are listed through the admin API.
        Then: the mock's ``get_weather`` tool is returned with its schema.
        """
        tools = _admin_get(
            ogx_admin_api_url=ogx_admin_api_url, path=f"/v1alpha/admin/connectors/{MOCK_MCP_CONNECTOR_ID}/tools"
        )
        tool_names = [tool["name"] for tool in tools["data"]]
        assert MOCK_MCP_TOOL_NAME in tool_names, f"{MOCK_MCP_TOOL_NAME} not listed: {tool_names}"

    @pytest.mark.tier2
    def test_responses_api_invokes_mcp_tool(self, ogx_client: OgxClient, mock_mcp_upstream: MockUpstream) -> None:
        """Verify the responses API executes an MCP tool server-side.

        Given: an OGX server with the builtin responses provider and a model
            (mock) that requests the get_weather tool.
        When: a response is created with an MCP tool referencing the mock MCP
            server.
        Then: OGX lists the MCP server's tools, invokes get_weather against
            the MCP server, and the tool output appears in the response.
        """
        response = ogx_client.responses.create(
            model=MOCK_MAAS_QUALIFIED_MODEL_ID,
            input=MCP_TOOL_PROMPT,
            tools=[
                {
                    "type": "mcp",
                    "server_label": MOCK_MCP_SERVER_LABEL,
                    "server_url": mock_mcp_upstream.service_url,
                    "allowed_tools": [MOCK_MCP_TOOL_NAME],
                }
            ],
        )
        assert response.status == "completed", f"Response not completed: {response.error}"

        output_items = response.output
        mcp_calls = [item for item in output_items if getattr(item, "type", None) == "mcp_call"]
        assert mcp_calls, f"No mcp_call in response output: {[getattr(i, 'type', None) for i in output_items]}"

        mcp_call = mcp_calls[0]
        assert mcp_call.name == MOCK_MCP_TOOL_NAME, f"Unexpected tool invoked: {mcp_call.name}"
        assert mcp_call.error is None, f"MCP tool call errored: {mcp_call.error}"
        assert MOCK_MCP_TOOL_RESULT_TEXT in str(mcp_call.output), f"Tool output missing: {mcp_call.output}"
