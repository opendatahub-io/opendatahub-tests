from typing import Self

import pytest
import structlog
import yaml
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.config_map import ConfigMap

from tests.model_registry.constants import DEFAULT_MODEL_CATALOG_CM
from tests.model_registry.mcp_servers.config.constants import EXPECTED_DEFAULT_MCP_CATALOG
from tests.model_registry.utils import execute_get_command

LOGGER = structlog.get_logger(name=__name__)
REQUIRED_SERVER_FIELDS: list[str] = ["name", "version", "description", "readme"]

pytestmark = [pytest.mark.install, pytest.mark.post_upgrade]


@pytest.mark.smoke
class TestDefaultMCPCatalogSourceConfigMap:
    """Tests for the default MCP catalog source ConfigMap entry."""

    def test_default_mcp_catalog_entry(
        self: Self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
    ):
        """Verify that model-catalog-default-sources ConfigMap contains the expected Red Hat MCP Servers entry."""
        configmap = ConfigMap(
            name=DEFAULT_MODEL_CATALOG_CM,
            client=admin_client,
            namespace=model_registry_namespace,
        )
        sources_data = yaml.safe_load(configmap.instance.data.get("sources.yaml", "{}") or "{}")
        mcp_catalogs = sources_data.get("mcp_catalogs", [])

        matching = [entry for entry in mcp_catalogs if entry.get("id") == EXPECTED_DEFAULT_MCP_CATALOG["id"]]
        assert matching, (
            f"Expected mcp_catalogs entry with id '{EXPECTED_DEFAULT_MCP_CATALOG['id']}' "
            f"not found in {DEFAULT_MODEL_CATALOG_CM} ConfigMap. Found entries: {mcp_catalogs}"
        )
        assert matching[0] == EXPECTED_DEFAULT_MCP_CATALOG, (
            f"MCP catalog entry does not match expected values.\n"
            f"Expected: {EXPECTED_DEFAULT_MCP_CATALOG}\n"
            f"Actual: {matching[0]}"
        )
        LOGGER.info(f"Found {len(matching)} MCP catalogs from default catalog")


@pytest.mark.tier1
class TestDefaultMCPCatalogSourceValidations:
    """Tests for the default MCP catalog source API validations."""

    def test_default_mcp_servers_loaded(
        self: Self,
        default_mcp_servers: dict,
    ):
        """Verify that the default MCP catalog returns a non-empty list of servers."""
        size = default_mcp_servers.get("size", 0)
        items = default_mcp_servers.get("items", [])
        LOGGER.info(f"Found {len(items)} MCP servers from default catalog (size={size})")
        assert size > 0, f"Expected size > 0 from default MCP catalog, but got {size}"
        assert len(items) > 0, "Expected at least one MCP server from the default catalog, but got none"

    def test_default_mcp_servers_required_fields(
        self: Self,
        default_mcp_servers: dict,
    ):
        """Verify that all default MCP servers contain required metadata fields."""
        errors = []
        for server in default_mcp_servers.get("items", []):
            server_name = server["name"]
            for field in REQUIRED_SERVER_FIELDS:
                if not server.get(field):
                    errors.append(f"Server '{server_name}' is missing required field '{field}'")
        assert not errors, "Required field validation failed:\n" + "\n".join(errors)

    def test_get_default_mcp_server_by_id(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        random_default_mcp_server: dict,
    ):
        """Verify that fetching an MCP server by id returns the same data as the list response."""
        server_id = random_default_mcp_server["id"]
        response = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers/{server_id}",
            headers=model_registry_rest_headers,
        )
        LOGGER.info(f"Fetched MCP server by id: {server_id}")
        assert response == random_default_mcp_server, (
            f"Server fetched by id does not match list response.\n"
            f"Expected: {random_default_mcp_server}\n"
            f"Actual: {response}"
        )

    def test_default_mcp_server_get_tools(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        default_mcp_servers: dict,
    ):
        """Verify that all default MCP servers have tools and each tool has required fields."""
        errors = []
        for server in default_mcp_servers.get("items", []):
            server_name = server["name"]
            server_id = server["id"]
            tool_count = server.get("toolCount", 0)
            if tool_count == 0:
                errors.append(f"Server '{server_name}' has toolCount=0")
                continue
            response = execute_get_command(
                url=f"{mcp_catalog_rest_urls[0]}mcp_servers/{server_id}/tools",
                headers=model_registry_rest_headers,
                params={"pageSize": 1000},
            )
            if response["size"] != tool_count:
                errors.append(f"Server '{server_name}': toolCount={tool_count} but got {response['size']} tools")
            for tool in response["items"]:
                missing_fields = [field for field in ["id", "name", "accessType", "description"] if not tool.get(field)]
                if missing_fields:
                    tool_name = tool.get("name", "<unnamed>")
                    errors.append(
                        f"Server '{server_name}' tool '{tool_name}' is missing fields: {', '.join(missing_fields)}"
                    )
        assert not errors, "Tool validation failed:\n" + "\n".join(errors)

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

    def test_get_default_mcp_server_tool_by_name(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        random_default_mcp_server: dict,
        random_default_mcp_server_tool: dict,
    ):
        """Verify that fetching a specific tool by name returns the same data as the tools list."""
        server_id = random_default_mcp_server["id"]
        tool_name = random_default_mcp_server_tool["name"]
        response = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers/{server_id}/tools/{tool_name}",
            headers=model_registry_rest_headers,
        )

        LOGGER.info(f"Fetched tool '{tool_name}' for MCP server: {random_default_mcp_server['name']}")
        assert response == random_default_mcp_server_tool, (
            f"Tool fetched by name does not match tools list response.\n"
            f"Expected: {random_default_mcp_server_tool}\n"
            f"Actual: {response}"
        )

    def test_default_mcp_server_tools_loaded(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        default_mcp_servers: dict,
    ):
        """Verify that default MCP server tools are correctly loaded when includeTools=true."""
        response = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
            headers=model_registry_rest_headers,
            params={"includeTools": "true", "toolLimit": "100"},
        )
        default_server_names = {server["name"] for server in default_mcp_servers.get("items", [])}
        default_servers = [server for server in response.get("items", []) if server["name"] in default_server_names]
        errors = []
        for server in default_servers:
            name = server["name"]
            tool_count = server.get("toolCount", 0)
            tools = server.get("tools", [])
            if not tools:
                errors.append(f"Server '{name}' has no tools with includeTools=true")
                continue
            if len(tools) != tool_count:
                errors.append(f"Server '{name}': toolCount={tool_count} but got {len(tools)} tools")
        assert not errors, "Tool loading validation failed:\n" + "\n".join(errors)


class TestDefaultMCPDisable:
    """Tests for verifying behavior when the default MCP catalog source is disabled."""

    def test_default_mcp_servers_disabled(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        disable_default_mcp_source: dict,
        default_mcp_servers: dict,
    ):
        """Verify that default MCP servers are not returned when the default source is disabled."""
        response = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
            headers=model_registry_rest_headers,
        )
        returned_server_ids = {server["source_id"] for server in response.get("items", [])}
        for server in default_mcp_servers.get("items", []):
            assert server["source_id"] not in returned_server_ids, (
                f"Default MCP server '{server['name']}' should not be present after disabling source"
            )
