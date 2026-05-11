from typing import Self

import pytest
import structlog
from kubernetes.dynamic.exceptions import ResourceNotFoundError

from tests.model_registry.mcp_servers.config.constants import (
    DEFAULT_MCP_LABEL,
    EXPECTED_COMMUNITY_MCP_CATALOG,
    EXPECTED_DEFAULT_MCP_CATALOG,
    EXPECTED_PARTNER_MCP_CATALOG,
    EXPECTED_PARTNER_MCP_LABEL_DEFINITION,
    EXPECTED_REDHAT_MCP_LABEL_DEFINITION,
    PARTNER_MCP_LABEL,
)
from tests.model_registry.utils import execute_get_command

LOGGER = structlog.get_logger(name=__name__)
REQUIRED_SERVER_FIELDS: list[str] = ["name", "version", "description", "readme"]

pytestmark = [pytest.mark.install, pytest.mark.post_upgrade]


@pytest.mark.smoke
class TestDefaultMCPCatalogSourceConfigMap:
    """Tests for the default MCP catalog source ConfigMap entry."""

    @pytest.mark.parametrize(
        "expected_catalog",
        [
            pytest.param(EXPECTED_DEFAULT_MCP_CATALOG, id="test_redhat_catalog"),
            pytest.param(EXPECTED_PARTNER_MCP_CATALOG, id="test_partner_catalog"),
            pytest.param(EXPECTED_COMMUNITY_MCP_CATALOG, id="test_community_catalog"),
        ],
    )
    def test_default_mcp_catalog_entry(
        self: Self,
        default_mcp_catalogs: list[dict],
        expected_catalog: dict,
    ):
        """Verify that default-catalog-sources ConfigMap contains the expected MCP catalog entry."""
        matching = [entry for entry in default_mcp_catalogs if entry.get("id") == expected_catalog["id"]]
        assert len(matching) == 1, (
            f"Expected exactly 1 mcp_catalogs entry with id '{expected_catalog['id']}', "
            f"found {len(matching)}: {matching}"
        )
        assert matching[0] == expected_catalog, (
            f"MCP catalog entry does not match expected values.\nExpected: {expected_catalog}\nActual: {matching[0]}"
        )

    @pytest.mark.parametrize(
        "expected_label",
        [
            pytest.param(EXPECTED_REDHAT_MCP_LABEL_DEFINITION, id="test_redhat_label"),
            pytest.param(EXPECTED_PARTNER_MCP_LABEL_DEFINITION, id="test_partner_label"),
        ],
    )
    def test_default_mcp_label_definition(
        self: Self,
        default_mcp_label_definitions: list[dict],
        expected_label: dict,
    ):
        """Verify that the default catalog sources ConfigMap contains the expected MCP label definition."""
        matching = [label for label in default_mcp_label_definitions if label.get("name") == expected_label["name"]]
        assert len(matching) == 1, (
            f"Expected exactly 1 label definition with name '{expected_label['name']}', "
            f"found {len(matching)}: {matching}"
        )
        assert matching[0] == expected_label, (
            f"Label definition does not match expected values.\nExpected: {expected_label}\nActual: {matching[0]}"
        )


@pytest.mark.tier1
@pytest.mark.parametrize(
    "mcp_servers_by_source",
    [
        pytest.param(DEFAULT_MCP_LABEL, id="test_redhat"),
        pytest.param(PARTNER_MCP_LABEL, id="test_partner"),
        pytest.param("null", id="test_community"),
    ],
    indirect=True,
)
class TestDefaultMCPCatalogSourceValidations:
    """Tests for the default MCP catalog source API validations."""

    def test_default_mcp_servers_loaded(
        self: Self,
        mcp_servers_by_source: dict,
    ):
        """Verify that the MCP catalog returns a non-empty list of servers for the given source label."""
        size = mcp_servers_by_source.get("size", 0)
        items = mcp_servers_by_source.get("items", [])
        LOGGER.info(f"Found {len(items)} MCP servers (size={size})")
        assert size > 0, f"Expected size > 0, but got {size}"
        assert len(items) > 0, "Expected at least one MCP server, but got none"

    def test_default_mcp_servers_required_fields(
        self: Self,
        mcp_servers_by_source: dict,
    ):
        """Verify that all MCP servers contain required metadata fields."""
        errors = []
        for server in mcp_servers_by_source.get("items", []):
            server_name = server["name"]
            for field in REQUIRED_SERVER_FIELDS:
                if not server.get(field):
                    errors.append(f"Server '{server_name}' is missing required field '{field}'")
        assert not errors, "Required field validation failed:\n" + "\n".join(errors)

    def test_get_default_mcp_server_by_id(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        mcp_servers_by_source: dict,
    ):
        """Verify that fetching an MCP server by id returns the same data as the list response."""
        server = mcp_servers_by_source["items"][0]
        server_id = server["id"]
        fetched = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers/{server_id}",
            headers=model_registry_rest_headers,
        )
        LOGGER.info(f"Fetched MCP server by id: {server_id}")
        assert fetched == server, (
            f"Server fetched by id does not match list response.\nExpected: {server}\nActual: {fetched}"
        )

    def test_default_mcp_server_get_tools(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        mcp_servers_by_source: dict,
    ):
        """Verify that all MCP servers have tools and each tool has required fields."""
        errors = []
        for server in mcp_servers_by_source.get("items", []):
            server_name = server["name"]
            server_id = server["id"]
            tool_count = server.get("toolCount", 0)
            if tool_count == 0:
                errors.append(f"Server '{server_name}' has toolCount=0")
                continue
            tools_response = execute_get_command(
                url=f"{mcp_catalog_rest_urls[0]}mcp_servers/{server_id}/tools",
                headers=model_registry_rest_headers,
                params={"pageSize": 1000},
            )
            if tools_response["size"] != tool_count:
                errors.append(f"Server '{server_name}': toolCount={tool_count} but got {tools_response['size']} tools")
            for tool in tools_response["items"]:
                missing_fields = [field for field in ["id", "name", "accessType", "description"] if not tool.get(field)]
                if missing_fields:
                    tool_name = tool.get("name", "<unnamed>")
                    errors.append(
                        f"Server '{server_name}' tool '{tool_name}' is missing fields: {', '.join(missing_fields)}"
                    )
        assert not errors, "Tool validation failed:\n" + "\n".join(errors)

    def test_mcp_server_by_id_tool_limit(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        mcp_servers_by_source: dict,
    ):
        """Verify toolLimit caps returned tools array (TC-API-021)."""
        tool_limit = 1
        server = mcp_servers_by_source["items"][0]
        response = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers/{server['id']}",
            headers=model_registry_rest_headers,
            params={"includeTools": "true", "toolLimit": str(tool_limit)},
        )
        tools = response.get("tools", [])
        assert len(tools) <= tool_limit, (
            f"Server '{server['name']}' returned {len(tools)} tools, expected at most {tool_limit}"
        )

    def test_mcp_servers_tool_limit(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        mcp_servers_by_source: dict,
    ):
        """Verify toolLimit caps returned tools array on the mcp_servers list endpoint."""
        tool_limit = 1
        response = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
            headers=model_registry_rest_headers,
            params={"includeTools": "true", "toolLimit": str(tool_limit), "pageSize": 1000},
        )
        for server in response.get("items", []):
            name = server["name"]
            tools = server.get("tools", [])
            assert len(tools) <= tool_limit, (
                f"Server '{name}' returned {len(tools)} tools, expected at most {tool_limit}"
            )

    @pytest.mark.tier3
    def test_tool_limit_exceeding_maximum(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        mcp_servers_by_source: dict,
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
        mcp_servers_by_source: dict,
    ):
        """Verify that fetching a specific tool by name returns the same data as the tools list."""
        server = next(
            (s for s in mcp_servers_by_source.get("items", []) if s.get("toolCount", 0) > 0),
            None,
        )
        assert server, "No server with tools found"
        server_id = server["id"]
        tools_response = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers/{server_id}/tools",
            headers=model_registry_rest_headers,
            params={"pageSize": 1000},
        )
        tool = tools_response["items"][0]
        tool_name = tool["name"]
        fetched = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers/{server_id}/tools/{tool_name}",
            headers=model_registry_rest_headers,
        )
        LOGGER.info(f"Fetched tool '{tool_name}' for MCP server: {server['name']}")
        assert fetched == tool, (
            f"Tool fetched by name does not match tools list response.\nExpected: {tool}\nActual: {fetched}"
        )

    def test_default_mcp_server_tools_pagination(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        mcp_servers_by_source: dict,
        mcp_server_with_multiple_tools: tuple[str, int],
    ):
        """Verify that tools endpoint supports pagination with pageSize and nextPageToken."""
        server_id, tool_count = mcp_server_with_multiple_tools
        tools_url = f"{mcp_catalog_rest_urls[0]}mcp_servers/{server_id}/tools"

        page_size = 1
        all_tool_names = []
        next_page_token = None

        for _ in range(tool_count):
            params: dict[str, str] = {"pageSize": str(page_size)}
            if next_page_token:
                params["nextPageToken"] = next_page_token

            response = execute_get_command(
                url=tools_url,
                headers=model_registry_rest_headers,
                params=params,
            )
            items = response.get("items", [])
            assert len(items) == page_size, f"Expected {page_size} tool per page, got {len(items)}"
            all_tool_names.extend(item["name"] for item in items)
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        assert len(set(all_tool_names)) == tool_count, (
            f"Expected {tool_count} unique tools, got {len(set(all_tool_names))}: {all_tool_names}"
        )

    def test_default_mcp_server_tools_loaded(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        mcp_servers_by_source: dict,
    ):
        """Verify that MCP server tools are correctly loaded when includeTools=true."""
        errors = []
        for server in mcp_servers_by_source.get("items", []):
            server_id = server["id"]
            name = server["name"]
            response = execute_get_command(
                url=f"{mcp_catalog_rest_urls[0]}mcp_servers/{server_id}",
                headers=model_registry_rest_headers,
                params={"includeTools": "true", "toolLimit": "100"},
            )
            tool_count = response.get("toolCount", 0)
            tools = response.get("tools", [])
            if not tools:
                errors.append(f"Server '{name}' has no tools with includeTools=true")
                continue
            if len(tools) != tool_count:
                errors.append(f"Server '{name}': toolCount={tool_count} but got {len(tools)} tools")
        assert not errors, "Tool loading validation failed:\n" + "\n".join(errors)


@pytest.mark.parametrize(
    "disable_default_mcp_source",
    [
        pytest.param(EXPECTED_DEFAULT_MCP_CATALOG, id="test_redhat"),
        pytest.param(EXPECTED_PARTNER_MCP_CATALOG, id="test_partner"),
        pytest.param(EXPECTED_COMMUNITY_MCP_CATALOG, id="test_community"),
    ],
    indirect=True,
)
class TestDefaultMCPDisable:
    """Tests for verifying behavior when a default MCP catalog source is disabled."""

    def test_default_mcp_servers_disabled(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        disable_default_mcp_source: str,
    ):
        """Verify that MCP servers from the disabled source are not returned."""
        response = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
            headers=model_registry_rest_headers,
            params={"pageSize": 1000},
        )
        returned_source_ids = {server["source_id"] for server in response.get("items", [])}
        assert disable_default_mcp_source not in returned_source_ids, (
            "Servers from disabled source should not be present, but source_id found in response"
        )
