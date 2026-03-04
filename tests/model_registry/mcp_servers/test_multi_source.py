from typing import Self

import pytest
from simple_logger.logger import get_logger

from tests.model_registry.mcp_servers.constants import (
    EXPECTED_ALL_MCP_SERVER_NAMES,
    EXPECTED_MCP_SOURCE_ID_MAP,
)
from tests.model_registry.utils import execute_get_command

LOGGER = get_logger(name=__name__)


@pytest.mark.usefixtures("mcp_multi_source_configmap_patch")
class TestMCPServerMultiSource:
    """RHOAIENG-51582: Tests for loading MCP servers from multiple YAML sources (TC-LOAD-002)."""

    def test_all_servers_from_multiple_sources_loaded(
        self: Self,
        mcp_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Verify that servers from all configured sources are loaded."""
        response = execute_get_command(
            url=f"{mcp_catalog_rest_url[0]}mcp_servers",
            headers=model_registry_rest_headers,
        )
        server_names = {server["name"] for server in response.get("items", [])}
        assert server_names == EXPECTED_ALL_MCP_SERVER_NAMES

    def test_servers_tagged_with_correct_source_id(
        self: Self,
        mcp_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Verify that each server is tagged with the correct source_id from its source."""
        response = execute_get_command(
            url=f"{mcp_catalog_rest_url[0]}mcp_servers",
            headers=model_registry_rest_headers,
        )
        for server in response.get("items", []):
            name = server["name"]
            expected_source = EXPECTED_MCP_SOURCE_ID_MAP[name]
            assert server.get("source_id") == expected_source, (
                f"Server '{name}' has source_id '{server.get('source_id')}', expected '{expected_source}'"
            )
