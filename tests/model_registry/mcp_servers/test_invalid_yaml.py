from typing import Self

import pytest
from simple_logger.logger import get_logger

from tests.model_registry.mcp_servers.constants import (
    EXPECTED_MCP_SERVER_NAMES,
    MCP_SERVERS_YAML_MALFORMED,
    MCP_SERVERS_YAML_MISSING_NAME,
)
from tests.model_registry.utils import execute_get_command

LOGGER = get_logger(name=__name__)


@pytest.mark.parametrize(
    "mcp_invalid_yaml_configmap_patch",
    [MCP_SERVERS_YAML_MALFORMED],
    indirect=True,
)
@pytest.mark.usefixtures("mcp_invalid_yaml_configmap_patch")
class TestMCPServerMalformedYAML:
    """RHOAIENG-51582: Tests for graceful handling of malformed YAML syntax (TC-LOAD-007)."""

    def test_valid_servers_loaded_despite_malformed_source(
        self: Self,
        mcp_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Verify that valid MCP servers from a healthy source are still loaded
        when another source contains malformed YAML."""
        response = execute_get_command(
            url=f"{mcp_catalog_rest_url[0]}mcp_servers",
            headers=model_registry_rest_headers,
        )
        server_names = {server["name"] for server in response["items"]}
        assert EXPECTED_MCP_SERVER_NAMES <= server_names, (
            f"Expected valid servers {EXPECTED_MCP_SERVER_NAMES} to be loaded, got {server_names}"
        )


@pytest.mark.parametrize(
    "mcp_invalid_yaml_configmap_patch",
    [MCP_SERVERS_YAML_MISSING_NAME],
    indirect=True,
)
@pytest.mark.usefixtures("mcp_invalid_yaml_configmap_patch")
class TestMCPServerMissingRequiredField:
    """RHOAIENG-51582: Tests for graceful handling of missing required fields (TC-LOAD-008)."""

    def test_valid_servers_loaded_despite_missing_name(
        self: Self,
        mcp_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Verify that valid MCP servers from a healthy source are still loaded
        when another source has a server entry missing the required name field."""
        response = execute_get_command(
            url=f"{mcp_catalog_rest_url[0]}mcp_servers",
            headers=model_registry_rest_headers,
        )
        server_names = {server["name"] for server in response["items"]}
        assert EXPECTED_MCP_SERVER_NAMES <= server_names, (
            f"Expected valid servers {EXPECTED_MCP_SERVER_NAMES} to be loaded, got {server_names}"
        )
