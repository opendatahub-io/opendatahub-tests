from typing import Self

import pytest
import structlog

from tests.model_registry.mcp_servers.config.constants import (
    EXPECTED_MCP_SERVER_NAMES,
    EXPECTED_MCP_SERVER_TIMESTAMPS,
    EXPECTED_MCP_SERVER_TOOL_COUNTS,
)

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures("mcp_servers_configmap_patch")
class TestMCPServerLoading:
    """Tests for loading Custom MCP servers from YAML into the catalog (TC-LOAD-001)."""

    def test_custom_mcp_servers(
        self: Self,
        mcp_servers_response: list[dict],
        default_mcp_servers: dict,
    ):
        """Verify MCP servers loaded with correct timestamps and tools excluded by default (TC-LOAD-001, TC-API-020)."""
        servers_by_name = {server["name"]: server for server in mcp_servers_response.get("items", [])}
        default_server_names = {server["name"] for server in default_mcp_servers.get("items", [])}
        assert default_server_names.issubset(set(servers_by_name)), (
            f"Missing default servers: {default_server_names - set(servers_by_name)}"
        )
        assert EXPECTED_MCP_SERVER_NAMES.issubset(set(servers_by_name)), (
            f"Missing expected servers: {EXPECTED_MCP_SERVER_NAMES - set(servers_by_name)}"
        )

    def test_custom_mcp_servers_metadata(
        self: Self,
        custom_mcp_servers: list[dict],
    ):
        """Verify custom MCP servers have correct timestamps and tool counts (TC-LOAD-001, TC-API-020)."""
        servers_by_name = {server["name"]: server for server in custom_mcp_servers}
        for name, server in servers_by_name.items():
            expected = EXPECTED_MCP_SERVER_TIMESTAMPS[name]
            assert server["createTimeSinceEpoch"] == expected["createTimeSinceEpoch"]
            assert server["lastUpdateTimeSinceEpoch"] == expected["lastUpdateTimeSinceEpoch"]
            assert "tools" not in server or server["tools"] is None, (
                f"Server '{name}' should not include tools by default"
            )
            assert server["toolCount"] == EXPECTED_MCP_SERVER_TOOL_COUNTS[name]
