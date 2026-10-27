"""Quickstart discovery tests for rhoai-mcp.

Verifies that the MCP server can read the Red Hat AI quickstart catalog,
which is published as an OCI artifact on Quay, and return it to an MCP client.
"""

import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from tests.rhoai_mcp.quickstarts.constants import (
    RHOAI_MCP_QUICKSTART_GOVERNED_TOOLS,
    RHOAI_MCP_QUICKSTART_NAME,
    RHOAI_MCP_QUICKSTART_OPEN_TOOLS,
    RHOAI_MCP_QUICKSTART_REQUIRED_ACTIONS,
    RHOAI_MCP_UNKNOWN_QUICKSTART_NAME,
)
from tests.rhoai_mcp.utils import parse_tool_result


@pytest.mark.asyncio
@pytest.mark.tier1
class TestRhoaiMcpQuickstartDiscovery:
    """Verify rhoai-mcp lists quickstarts from the Red Hat AI quickstart registry."""

    async def test_list_tools_includes_all_quickstart_tools(
        self,
        rhoai_mcp_transport: StreamableHttpTransport,
    ) -> None:
        """Given rhoai-mcp is deployed and the caller is a cluster administrator
        When tools/list is called via the FastMCP client
        Then the response includes every quickstart tool
        """
        async with Client(rhoai_mcp_transport) as client:
            tools = await client.list_tools()
            tool_names = {tool.name for tool in tools}

        expected = set(RHOAI_MCP_QUICKSTART_OPEN_TOOLS) | set(RHOAI_MCP_QUICKSTART_GOVERNED_TOOLS)
        missing = expected - tool_names
        assert not missing, f"Expected quickstart tools not found: {missing}. Got: {sorted(tool_names)}"

    async def test_list_quickstarts_includes_known_quickstart(
        self,
        rhoai_mcp_transport: StreamableHttpTransport,
    ) -> None:
        """Given rhoai-mcp is deployed and can reach the quickstart registry
        When list_quickstarts is called via the FastMCP client
        Then the catalog includes the known quickstart with a latest manifest version
        """
        async with Client(rhoai_mcp_transport) as client:
            result = await client.call_tool(name="list_quickstarts", arguments={})
            data = parse_tool_result(result=result)

        assert "error" not in data, f"list_quickstarts returned an error: {data['error']}"
        quickstarts = {item["name"]: item for item in data["items"]}
        assert RHOAI_MCP_QUICKSTART_NAME in quickstarts, (
            f"Expected quickstart '{RHOAI_MCP_QUICKSTART_NAME}' not found. Got: {sorted(quickstarts)}"
        )
        assert quickstarts[RHOAI_MCP_QUICKSTART_NAME]["latest_manifest_version"], (
            f"Quickstart '{RHOAI_MCP_QUICKSTART_NAME}' has no latest manifest version"
        )

    async def test_get_quickstart_manifest_returns_deployable_manifest(
        self,
        rhoai_mcp_transport: StreamableHttpTransport,
    ) -> None:
        """Given rhoai-mcp is deployed and can reach the quickstart registry
        When get_quickstart_manifest is called for the known quickstart
        Then the manifest supports the lifecycle actions and pins its installer image by digest
        """
        async with Client(rhoai_mcp_transport) as client:
            result = await client.call_tool(
                name="get_quickstart_manifest",
                arguments={"name": RHOAI_MCP_QUICKSTART_NAME},
            )
            data = parse_tool_result(result=result)

        assert "error" not in data, f"get_quickstart_manifest returned an error: {data['error']}"
        assert data["name"] == RHOAI_MCP_QUICKSTART_NAME
        missing = set(RHOAI_MCP_QUICKSTART_REQUIRED_ACTIONS) - set(data["supported_actions"])
        assert not missing, f"Manifest is missing actions {missing}. Got: {data['supported_actions']}"
        assert "@sha256:" in data["installer_image"], (
            f"Installer image is not pinned by digest: {data['installer_image']}"
        )

    async def test_get_quickstart_manifest_unknown_name_returns_error(
        self,
        rhoai_mcp_transport: StreamableHttpTransport,
    ) -> None:
        """Given rhoai-mcp is deployed and can reach the quickstart registry
        When get_quickstart_manifest is called for a quickstart that is not in the catalog
        Then the tool returns a not-found error instead of a manifest
        """
        async with Client(rhoai_mcp_transport) as client:
            result = await client.call_tool(
                name="get_quickstart_manifest",
                arguments={"name": RHOAI_MCP_UNKNOWN_QUICKSTART_NAME},
            )
            data = parse_tool_result(result=result)

        assert data.get("error", "").startswith(f"Quickstart '{RHOAI_MCP_UNKNOWN_QUICKSTART_NAME}' not found"), (
            f"Expected a not-found error, got: {data}"
        )
