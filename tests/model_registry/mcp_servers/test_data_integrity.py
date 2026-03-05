import random
from typing import Any, Self

import pytest
from simple_logger.logger import get_logger

from tests.model_registry.mcp_servers.constants import (
    EXPECTED_MCP_SERVER_CUSTOM_PROPERTIES,
    EXPECTED_MCP_SERVER_NAMES,
    EXPECTED_MCP_SERVER_PROVIDERS,
    EXPECTED_MCP_SERVER_TIMESTAMPS,
    EXPECTED_MCP_SERVER_TOOL_COUNTS,
    EXPECTED_MCP_SERVER_TOOLS,
)
from tests.model_registry.utils import execute_get_command

LOGGER = get_logger(name=__name__)


@pytest.mark.usefixtures("mcp_servers_configmap_patch")
class TestMCPServerLoading:
    """RHOAIENG-51582: Tests for loading MCP servers from YAML into the catalog (TC-LOAD-001)."""

    def test_mcp_servers_loaded(
        self: Self,
        mcp_servers_response: dict[str, Any],
    ):
        """Verify that all MCP servers and their providers are loaded from YAML."""
        servers_by_name = {server["name"]: server for server in mcp_servers_response["items"]}
        assert set(servers_by_name) == EXPECTED_MCP_SERVER_NAMES
        actual_providers = {name: server["provider"] for name, server in servers_by_name.items()}
        assert actual_providers == EXPECTED_MCP_SERVER_PROVIDERS
        for name, server in servers_by_name.items():
            expected = EXPECTED_MCP_SERVER_TIMESTAMPS[name]
            assert server["createTimeSinceEpoch"] == expected["createTimeSinceEpoch"]
            assert server["lastUpdateTimeSinceEpoch"] == expected["lastUpdateTimeSinceEpoch"]

    @pytest.mark.xfail(
        reason="RHOAIENG-51765: Tool name returned as {server}@{version}:{tool} instead of YAML-defined name"
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

    @pytest.mark.xfail(reason="RHOAIENG-51764: toolCount is 0 when not passing includeTools=true")
    def test_mcp_server_tool_count_without_include(
        self: Self,
        mcp_servers_response: dict[str, Any],
    ):
        """Verify that toolCount reflects actual tools even when tools are not included."""
        for server in mcp_servers_response.get("items", []):
            name = server["name"]
            expected_count = EXPECTED_MCP_SERVER_TOOL_COUNTS[name]
            actual_count = server.get("toolCount", 0)
            assert actual_count == expected_count, (
                f"Server '{name}': expected toolCount {expected_count}, got {actual_count}"
            )

    def test_mcp_server_get_by_id(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        mcp_servers_response: dict[str, Any],
    ):
        """Verify that an MCP server can be retrieved by ID."""
        server = random.choice(seq=mcp_servers_response["items"])
        single_server = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers/{server['id']}",
            headers=model_registry_rest_headers,
        )
        assert single_server["name"] == server["name"], (
            f"Expected server name '{server['name']}' for ID '{server['id']}', got '{single_server['name']}'"
        )

    def test_mcp_server_custom_properties(
        self: Self,
        mcp_servers_response: dict[str, Any],
    ):
        """Verify that customProperties are correctly loaded from YAML (TC-LOAD-010)."""
        servers_by_name = {server["name"]: server for server in mcp_servers_response["items"]}
        for name, expected in EXPECTED_MCP_SERVER_CUSTOM_PROPERTIES.items():
            assert servers_by_name[name]["customProperties"] == expected
