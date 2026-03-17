from typing import Any, Self

import pytest
from simple_logger.logger import get_logger

from tests.model_registry.mcp_servers.constants import (
    EXPECTED_MCP_SERVER_NAMES,
    EXPECTED_MCP_SERVER_TIMESTAMPS,
    EXPECTED_MCP_SERVER_TOOLS,
)
from tests.model_registry.utils import execute_get_command

LOGGER = get_logger(name=__name__)


@pytest.mark.usefixtures("mcp_servers_configmap_patch")
class TestMCPServerLoading:
    """Tests for loading MCP servers from YAML into the catalog (TC-LOAD-001)."""

    def test_mcp_servers_loaded(
        self: Self,
        mcp_servers_response: dict[str, Any],
    ):
        """Verify that all MCP servers are loaded from YAML with correct timestamps."""
        servers_by_name = {server["name"]: server for server in mcp_servers_response["items"]}
        assert set(servers_by_name) == EXPECTED_MCP_SERVER_NAMES
        for name, server in servers_by_name.items():
            expected = EXPECTED_MCP_SERVER_TIMESTAMPS[name]
            assert server["createTimeSinceEpoch"] == expected["createTimeSinceEpoch"]
            assert server["lastUpdateTimeSinceEpoch"] == expected["lastUpdateTimeSinceEpoch"]

    def test_mcp_server_tools_loaded(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Verify that MCP server tools are correctly loaded when includeTools=true."""
        response = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
            headers=model_registry_rest_headers,
            params={"includeTools": "true"},
        )
        for server in response.get("items", []):
            name = server["name"]
            expected_tool_names = EXPECTED_MCP_SERVER_TOOLS[name]
            assert server["toolCount"] == len(expected_tool_names)
            actual_tool_names = [tool["name"] for tool in server["tools"]]
            assert sorted(actual_tool_names) == sorted(expected_tool_names)
