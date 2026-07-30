"""RBAC-based tool filtering tests for rhoai-mcp.

Verifies that the MCP server correctly filters tools based on the
authenticated user's Kubernetes RBAC permissions, using
SubjectAccessReview checks at both ListToolsRequest and CallToolRequest time.
"""

import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.exceptions import ToolError

from tests.rhoai_mcp.constants import (
    RHOAI_MCP_EXPECTED_CATALOG_TOOLS,
    RHOAI_MCP_INFERENCE_DEPLOY_TOOLS,
    RHOAI_MCP_INFERENCE_READ_TOOLS,
    RHOAI_MCP_INFERENCE_RESTRICTED_TOOLS,
    RHOAI_MCP_NAMESPACE,
)


@pytest.mark.asyncio
@pytest.mark.tier1
class TestRhoaiMcpRbac:
    """Verify rhoai-mcp filters MCP tools based on Kubernetes RBAC permissions.

    The two personas (reader / deployer) are intentionally minimal:
    a real deployer would typically also have delete and other verbs.
    These narrow permission sets exist only to demonstrate that the
    SAR-based tool filtering correctly shows or hides MCP tools
    depending on which Kubernetes RBAC verbs the caller holds.
    """

    async def test_reader_sees_read_tools_only(
        self,
        rbac_reader_transport: StreamableHttpTransport,
    ) -> None:
        """Given a user with get/list access to InferenceServices and ServingRuntimes
        When tools/list is called via an MCP client authenticated as that user
        Then the response includes read-only inference tools and catalog tools
        And the response excludes tools that require create or delete permissions
        """
        async with Client(rbac_reader_transport) as client:
            tools = await client.list_tools()
            tool_names = {tool.name for tool in tools}

            expected_visible = set(RHOAI_MCP_INFERENCE_READ_TOOLS) | set(RHOAI_MCP_EXPECTED_CATALOG_TOOLS)
            missing = expected_visible - tool_names
            assert not missing, f"Read/catalog tools not visible to reader: {missing}"

            expected_hidden = set(RHOAI_MCP_INFERENCE_DEPLOY_TOOLS) | set(RHOAI_MCP_INFERENCE_RESTRICTED_TOOLS)
            leaked = expected_hidden & tool_names
            assert not leaked, f"Write tools should be hidden from reader: {leaked}"

    async def test_reader_cannot_call_deploy_tool(
        self,
        rbac_reader_transport: StreamableHttpTransport,
    ) -> None:
        """Given a user with only get/list access to InferenceServices
        When tools/call is invoked for deploy_model by that user
        Then the server rejects the call with an error

        Note: deploy_model is not advertised to the Reader Persona
        yet invoked manually to prove RBAC checks are enforced in MCP
        """
        async with Client(rbac_reader_transport) as client:
            with pytest.raises(ToolError, match=r"deploy_model.*not permitted for the current user"):
                await client.call_tool(
                    name="deploy_model",
                    arguments={
                        "name": "rbac-denied-test",
                        "namespace": RHOAI_MCP_NAMESPACE,
                        "runtime": "vllm-runtime",
                        "model_format": "vLLM",
                        "storage_uri": "hf://instructlab/granite-7b-lab",
                    },
                )

    async def test_deployer_sees_read_and_deploy_tools(
        self,
        rbac_deployer_transport: StreamableHttpTransport,
    ) -> None:
        """Given a user with read+create on ISVC, read on SR, read+create on PVCs/Secrets
        When tools/list is called via an MCP client authenticated as that user
        Then the response includes read-only inference tools, deploy/prepare tools, and catalog tools
        And the response still excludes delete and create serving runtime tools
        """
        async with Client(rbac_deployer_transport) as client:
            tools = await client.list_tools()
            tool_names = {tool.name for tool in tools}

            expected_visible = (
                set(RHOAI_MCP_INFERENCE_READ_TOOLS)
                | set(RHOAI_MCP_INFERENCE_DEPLOY_TOOLS)
                | set(RHOAI_MCP_EXPECTED_CATALOG_TOOLS)
            )
            missing = expected_visible - tool_names
            assert not missing, f"Expected tools not visible to deployer: {missing}"

            expected_hidden = set(RHOAI_MCP_INFERENCE_RESTRICTED_TOOLS)
            leaked = expected_hidden & tool_names
            assert not leaked, f"Delete/create-runtime tools should be hidden from deployer: {leaked}"
