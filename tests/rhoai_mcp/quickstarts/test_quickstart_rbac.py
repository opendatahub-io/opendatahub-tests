"""RBAC-based tool filtering tests for the rhoai-mcp quickstart tools.

Verifies that quickstart discovery tools, which only read the public quickstart
registry, are open to any authenticated user, while tools that create or inspect
installer Jobs are hidden from, and rejected for, users without those permissions.
"""

import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.exceptions import ToolError

from tests.rhoai_mcp.quickstarts.constants import (
    RHOAI_MCP_QUICKSTART_GOVERNED_TOOLS,
    RHOAI_MCP_QUICKSTART_NAME,
    RHOAI_MCP_QUICKSTART_OPEN_TOOLS,
)


@pytest.mark.asyncio
@pytest.mark.tier1
class TestRhoaiMcpQuickstartRbac:
    """Verify rhoai-mcp filters quickstart tools based on Kubernetes RBAC permissions.

    The reader persona can only get/list InferenceServices and ServingRuntimes, so it
    holds none of the Job, Secret, Pod or Namespace permissions the quickstart action
    tools require.
    """

    async def test_reader_sees_discovery_tools_only(
        self,
        rbac_reader_transport: StreamableHttpTransport,
    ) -> None:
        """Given a user without permissions on Jobs, Secrets, Pods or Namespaces
        When tools/list is called via an MCP client authenticated as that user
        Then the response includes the quickstart discovery tools
        And the response excludes the quickstart action, status and log tools
        """
        async with Client(rbac_reader_transport) as client:
            tools = await client.list_tools()
            tool_names = {tool.name for tool in tools}

        missing = set(RHOAI_MCP_QUICKSTART_OPEN_TOOLS) - tool_names
        assert not missing, f"Quickstart discovery tools not visible to reader: {missing}"
        leaked = set(RHOAI_MCP_QUICKSTART_GOVERNED_TOOLS) & tool_names
        assert not leaked, f"Quickstart action tools should be hidden from reader: {leaked}"

    async def test_reader_cannot_call_run_quickstart_action(
        self,
        rbac_reader_transport: StreamableHttpTransport,
    ) -> None:
        """Given a user without permission to create installer Jobs
        When tools/call is invoked for run_quickstart_action by that user
        Then the server rejects the call without running the action

        Note: run_quickstart_action is not advertised to the reader persona,
        yet is invoked manually to prove RBAC checks are enforced at call time.
        """
        async with Client(rbac_reader_transport) as client:
            with pytest.raises(ToolError, match=r"run_quickstart_action.*not permitted for the current user"):
                await client.call_tool(
                    name="run_quickstart_action",
                    arguments={"name": RHOAI_MCP_QUICKSTART_NAME, "action": "CHECK_PRE_REQS"},
                )
