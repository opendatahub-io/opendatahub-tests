from typing import Self

import pytest
import structlog
from kubernetes.dynamic import DynamicClient

from tests.model_registry.mcp_servers.config.constants import (
    EXPECTED_MCP_SERVER_NAMES,
    MCP_CATALOG_INVALID_SOURCE_ID,
    MCP_CATALOG_INVALID_SOURCE_NAME,
    MCP_SERVERS_YAML_INVALID_CATALOG_PATH,
    MCP_SERVERS_YAML_MALFORMED,
    MCP_SERVERS_YAML_MISSING_NAME,
)
from tests.model_registry.utils import execute_get_command, get_model_catalog_pod

LOGGER = structlog.get_logger(name=__name__)

CATALOG_CONTAINER = "catalog"


@pytest.mark.parametrize(
    "mcp_invalid_yaml_configmap_patch, expected_log_error",
    [
        pytest.param(
            MCP_SERVERS_YAML_MALFORMED,
            f"{MCP_SERVERS_YAML_INVALID_CATALOG_PATH}: error parsing YAML",
            id="malformed_yaml",
        ),
        pytest.param(
            MCP_SERVERS_YAML_MISSING_NAME,
            "base_name cannot be empty",
            id="missing_name",
        ),
    ],
    indirect=["mcp_invalid_yaml_configmap_patch"],
)
@pytest.mark.usefixtures("mcp_invalid_yaml_configmap_patch")
class TestMCPServerInvalidYAML:
    """
    Tests for graceful handling of invalid YAML sources (TC-LOAD-007, TC-LOAD-008)."""

    def test_valid_servers_loaded_despite_invalid_source(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        expected_log_error: str,
    ):
        """Verify that valid MCP servers from a healthy source are still loaded
        when another source contains invalid YAML."""
        response = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
            headers=model_registry_rest_headers,
            params={"pageSize": 1000},
        )
        server_names = {server["name"] for server in response["items"]}
        assert EXPECTED_MCP_SERVER_NAMES.issubset(server_names), (
            f"Expected {EXPECTED_MCP_SERVER_NAMES} to be a subset of {server_names}"
        )

    def test_invalid_source_error_logged(
        self: Self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        expected_log_error: str,
    ):
        """Verify that pod logs contain the expected error for the invalid YAML source."""
        pod = get_model_catalog_pod(client=admin_client, model_registry_namespace=model_registry_namespace)[0]
        log = pod.log(container=CATALOG_CONTAINER)
        source_error = (
            f"Error loading servers from source {MCP_CATALOG_INVALID_SOURCE_NAME}: "
            f"all MCP servers failed to load from source {MCP_CATALOG_INVALID_SOURCE_ID}"
        )
        assert source_error in log, f"Expected '{source_error}' in pod logs"
        assert expected_log_error in log, f"Expected '{expected_log_error}' in pod logs"
