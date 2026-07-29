"""MCP capability discovery tests for rhoai-mcp.

Verifies that the MCP server correctly advertises its tools, resources,
and prompts through the standard MCP protocol using a FastMCP client
over Streamable HTTP transport.
"""

import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from tests.rhoai_mcp.constants import (
    RHOAI_MCP_EXPECTED_CATALOG_TOOLS,
    RHOAI_MCP_EXPECTED_PROMPTS,
    RHOAI_MCP_EXPECTED_SERVING_TOOLS,
)


@pytest.mark.asyncio
@pytest.mark.tier1  # Analogous to tests/ai_safety/evalhub/mcp/ which validates MCP capabilities.
class TestRhoaiMcpCapabilities:
    """Verify rhoai-mcp advertises expected MCP tools, resources, and prompts."""

    async def test_server_advertises_capabilities(
        self,
        rhoai_mcp_transport: StreamableHttpTransport,
    ) -> None:
        """Given rhoai-mcp is deployed and healthy
        When an authenticated MCP client initializes a session
        Then the server advertises tools, resources, and prompts capabilities
        """
        async with Client(rhoai_mcp_transport) as client:
            caps = client.initialize_result.capabilities
            assert caps.tools is not None, "Server did not advertise tools capability"
            assert caps.resources is not None, "Server did not advertise resources capability"
            assert caps.prompts is not None, "Server did not advertise prompts capability"

    async def test_list_tools_includes_serving_and_catalog_tools(
        self,
        rhoai_mcp_transport: StreamableHttpTransport,
    ) -> None:
        """Given rhoai-mcp is deployed and healthy
        When tools/list is called via the FastMCP client
        Then the response includes Model Serving and Model Catalog tools
        """
        async with Client(rhoai_mcp_transport) as client:
            tools = await client.list_tools()
            tool_names = {tool.name for tool in tools}
            expected = set(RHOAI_MCP_EXPECTED_SERVING_TOOLS) | set(RHOAI_MCP_EXPECTED_CATALOG_TOOLS)
            missing = expected - tool_names
            assert not missing, f"Expected tools not found: {missing}. Got: {sorted(tool_names)}"

    async def test_all_tools_have_descriptions(
        self,
        rhoai_mcp_transport: StreamableHttpTransport,
    ) -> None:
        """Given rhoai-mcp tools are listed
        When each tool's metadata is inspected
        Then every tool has a non-empty description
        """
        async with Client(rhoai_mcp_transport) as client:
            tools = await client.list_tools()
            assert tools, "Expected at least one tool"
            for tool in tools:
                assert tool.description and tool.description.strip(), f"Tool '{tool.name}' is missing a description"

    async def test_all_tools_have_input_schemas(
        self,
        rhoai_mcp_transport: StreamableHttpTransport,
    ) -> None:
        """Given rhoai-mcp tools are listed
        When each tool's input schema is inspected
        Then every tool has a valid JSON Schema with a type field
        """
        async with Client(rhoai_mcp_transport) as client:
            tools = await client.list_tools()
            assert tools, "Expected at least one tool"
            for tool in tools:
                schema = tool.inputSchema
                assert isinstance(schema, dict), f"Tool '{tool.name}' has no inputSchema"
                assert "type" in schema, f"Tool '{tool.name}' inputSchema missing 'type': {schema}"

    async def test_list_prompts_includes_expected_prompts(
        self,
        rhoai_mcp_transport: StreamableHttpTransport,
    ) -> None:
        """Given rhoai-mcp is deployed and healthy
        When prompts/list is called via the FastMCP client
        Then the response includes at least the representative expected prompts
        """
        async with Client(rhoai_mcp_transport) as client:
            prompts = await client.list_prompts()
            prompt_names = {prompt.name for prompt in prompts}
            expected = set(RHOAI_MCP_EXPECTED_PROMPTS)
            missing = expected - prompt_names
            assert not missing, f"Expected prompts not found: {missing}. Got: {sorted(prompt_names)}"

    async def test_all_prompts_have_descriptions(
        self,
        rhoai_mcp_transport: StreamableHttpTransport,
    ) -> None:
        """Given rhoai-mcp prompts are listed
        When each prompt's metadata is inspected
        Then every prompt has a non-empty description
        """
        async with Client(rhoai_mcp_transport) as client:
            prompts = await client.list_prompts()
            assert prompts, "Expected at least one prompt"
            for prompt in prompts:
                assert prompt.description and prompt.description.strip(), (
                    f"Prompt '{prompt.name}' is missing a description"
                )
