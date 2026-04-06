from typing import Self

import pytest
import structlog
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.resource import ResourceEditor

from tests.model_registry.mcp_servers.config.constants import (
    EXPECTED_ALL_MCP_SERVER_NAMES,
    EXPECTED_MCP_SOURCE_ID_MAP,
    MCP_CATALOG_SOURCE2_ID,
)
from tests.model_registry.mcp_servers.config.utils import get_mcp_catalog_sources
from tests.model_registry.utils import (
    execute_get_command,
    wait_for_mcp_catalog_api,
    wait_for_model_catalog_pod_ready_after_deletion,
)

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures("mcp_multi_source_configmap_patch")
class TestMCPServerMultiSource:
    """Tests for loading MCP servers from multiple YAML sources (TC-LOAD-002)."""

    def test_all_servers_from_multiple_sources_loaded(
        self: Self,
        custom_mcp_servers: list[dict],
    ):
        """Verify that servers from all configured sources are loaded."""
        server_names = {server["name"] for server in custom_mcp_servers}
        assert server_names == EXPECTED_ALL_MCP_SERVER_NAMES

    def test_servers_tagged_with_correct_source_id(
        self: Self,
        custom_mcp_servers: list[dict],
    ):
        """Verify that each server is tagged with the correct source_id from its source."""
        for server in custom_mcp_servers:
            name = server["name"]
            expected_source = EXPECTED_MCP_SOURCE_ID_MAP[name]
            assert server.get("source_id") == expected_source, (
                f"Server '{name}' has source_id '{server.get('source_id')}', expected '{expected_source}'"
            )

    @pytest.mark.parametrize(
        "cleanup_action",
        [
            pytest.param("disable", id="disable_source"),
            pytest.param("remove", id="remove_source"),
        ],
    )
    def test_source_cleanup_removes_servers(
        self: Self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        default_mcp_servers: dict,
        custom_mcp_servers: list[dict],
        cleanup_action: str,
    ):
        """TC-LOAD-011/012: Verify that disabling or removing a source removes its servers from the catalog."""
        catalog_config_map, current_data = get_mcp_catalog_sources(
            admin_client=admin_client, model_registry_namespace=model_registry_namespace
        )

        if cleanup_action == "disable":
            for source in current_data.get("mcp_catalogs", []):
                if source["id"] == MCP_CATALOG_SOURCE2_ID:
                    source["enabled"] = False
                    break
        else:
            current_data["mcp_catalogs"] = [
                source for source in current_data.get("mcp_catalogs", []) if source["id"] != MCP_CATALOG_SOURCE2_ID
            ]

        patches = {"data": {"sources.yaml": yaml.dump(current_data, default_flow_style=False)}}

        with ResourceEditor(patches={catalog_config_map: patches}):
            wait_for_model_catalog_pod_ready_after_deletion(
                client=admin_client, model_registry_namespace=model_registry_namespace
            )
            wait_for_mcp_catalog_api(url=mcp_catalog_rest_urls[0], headers=model_registry_rest_headers)

            response = execute_get_command(
                url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
                headers=model_registry_rest_headers,
            )
            remaining_names = {server["name"] for server in response["items"]}
            assert MCP_CATALOG_SOURCE2_ID not in remaining_names, (
                f"Expected  {MCP_CATALOG_SOURCE2_ID} not to be present after {cleanup_action} of source2, "
                f"got {remaining_names}"
            )

        wait_for_model_catalog_pod_ready_after_deletion(
            client=admin_client, model_registry_namespace=model_registry_namespace
        )
        wait_for_mcp_catalog_api(url=mcp_catalog_rest_urls[0], headers=model_registry_rest_headers)
