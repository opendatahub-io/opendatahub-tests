from typing import Any, Self

import pytest
import structlog
from kubernetes.dynamic.exceptions import ResourceNotFoundError

from tests.model_registry.mcp_servers.constants import (
    EXPECTED_MCP_SERVER_NAMES,
    EXPECTED_MCP_SERVER_TIMESTAMPS,
    EXPECTED_MCP_SERVER_TOOL_COUNTS,
    EXPECTED_MCP_SERVER_TOOLS,
)
from tests.model_registry.utils import execute_get_command

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures("mcp_servers_configmap_patch")
class TestMCPServerLoading:
    """Tests for loading MCP servers from YAML into the catalog (TC-LOAD-001)."""

    def test_mcp_servers_loaded(
        self: Self,
        mcp_servers_response: dict[str, Any],
    ):
        """Verify MCP servers loaded with correct timestamps and tools excluded by default (TC-LOAD-001, TC-API-020)."""
        servers_by_name = {server["name"]: server for server in mcp_servers_response["items"]}
        assert set(servers_by_name) == EXPECTED_MCP_SERVER_NAMES
        for name, server in servers_by_name.items():
            expected = EXPECTED_MCP_SERVER_TIMESTAMPS[name]
            assert server["createTimeSinceEpoch"] == expected["createTimeSinceEpoch"]
            assert server["lastUpdateTimeSinceEpoch"] == expected["lastUpdateTimeSinceEpoch"]
            assert "tools" not in server or server["tools"] is None, (
                f"Server '{name}' should not include tools by default"
            )
            assert server["toolCount"] == EXPECTED_MCP_SERVER_TOOL_COUNTS[name]

    def test_mcp_server_tool_limit(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Verify toolLimit caps returned tools array (TC-API-021)."""
        tool_limit = 1
        response = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
            headers=model_registry_rest_headers,
            params={"includeTools": "true", "toolLimit": str(tool_limit)},
        )
        for server in response.get("items", []):
            name = server["name"]
            assert len(server["tools"]) <= tool_limit, (
                f"Server '{name}' returned {len(server['tools'])} tools, expected at most {tool_limit}"
            )

    @pytest.mark.tier3
    def test_tool_limit_exceeding_maximum(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Verify toolLimit exceeding maximum (100) is rejected (TC-API-023)."""
        with pytest.raises(ResourceNotFoundError):
            execute_get_command(
                url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
                headers=model_registry_rest_headers,
                params={"includeTools": "true", "toolLimit": "101"},
            )

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
