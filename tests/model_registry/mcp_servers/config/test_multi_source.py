from typing import Any, Self

import pytest
from simple_logger.logger import get_logger

from tests.model_registry.mcp_servers.constants import (
    EXPECTED_ALL_MCP_SERVER_NAMES,
    EXPECTED_MCP_SOURCE_ID_MAP,
)

LOGGER = get_logger(name=__name__)


@pytest.mark.usefixtures("mcp_multi_source_configmap_patch")
class TestMCPServerMultiSource:
    """Tests for loading MCP servers from multiple YAML sources (TC-LOAD-002)."""

    def test_all_servers_from_multiple_sources_loaded(
        self: Self,
        mcp_servers_response: dict[str, Any],
    ):
        """Verify that servers from all configured sources are loaded."""
        server_names = {server["name"] for server in mcp_servers_response.get("items", [])}
        assert server_names == EXPECTED_ALL_MCP_SERVER_NAMES

    def test_servers_tagged_with_correct_source_id(
        self: Self,
        mcp_servers_response: dict[str, Any],
    ):
        """Verify that each server is tagged with the correct source_id from its source."""
        for server in mcp_servers_response.get("items", []):
            name = server["name"]
            expected_source = EXPECTED_MCP_SOURCE_ID_MAP[name]
            assert server.get("source_id") == expected_source, (
                f"Server '{name}' has source_id '{server.get('source_id')}', expected '{expected_source}'"
            )
