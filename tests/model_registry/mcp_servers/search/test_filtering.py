from typing import Self

import pytest
import structlog

from tests.model_registry.utils import execute_get_command

REDHAT_PROVIDER: str = "Red Hat"
LOGGER = structlog.get_logger(name=__name__)


class TestMCPServerFiltering:
    """Tests for MCP server filterQuery functionality."""

    @pytest.mark.parametrize(
        "filter_query, field_check",
        [
            pytest.param(
                f"provider='{REDHAT_PROVIDER}'",
                ("provider", REDHAT_PROVIDER),
                id="by_provider",
            ),
            pytest.param("license='Apache 2.0'", ("license", "Apache 2.0"), id="by_license"),
        ],
    )
    def test_filter_by_field(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        default_mcp_servers: dict,
        filter_query: str,
        field_check: tuple[str, str],
    ):
        """TC-API-003, TC-API-009: Test filtering MCP servers by provider and license."""
        response = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
            headers=model_registry_rest_headers,
            params={"filterQuery": filter_query},
        )
        items = response.get("items", [])
        expected_names = {
            server["name"]
            for server in default_mcp_servers.get("items", [])
            if server.get(field_check[0]) == field_check[1]
        }
        assert len(items) >= len(expected_names), (
            f"Expected at least {len(expected_names)} server(s) for '{filter_query}', got {len(items)}"
        )
        returned_names = {item["name"] for item in items}
        assert expected_names <= returned_names, (
            f"Expected default servers {expected_names} in results, got {returned_names}"
        )
        for item in items:
            assert item[field_check[0]] == field_check[1], (
                f"Server '{item['name']}' has {field_check[0]}='{item[field_check[0]]}', expected '{field_check[1]}'"
            )

    def test_filter_options(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """TC-API-026: Test that filter_options endpoint returns available filter fields."""
        url = f"{mcp_catalog_rest_urls[0]}mcp_servers/filter_options"
        LOGGER.info(f"Requesting filter_options from: {url}")

        response = execute_get_command(
            url=url,
            headers=model_registry_rest_headers,
        )
        LOGGER.info(f"filter_options full response: {response}")

        expected_filters = {"transports", "license", "provider", "architecture.array_value", "publishedDate"}
        assert expected_filters == set(response["filters"].keys())

    def test_pagination_with_filters(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """TC-API-032: Test that pagination works correctly with filterQuery."""
        base_url = f"{mcp_catalog_rest_urls[0]}mcp_servers"
        filter_query = "license='Apache 2.0'"

        # First page
        response = execute_get_command(
            url=base_url,
            headers=model_registry_rest_headers,
            params={"filterQuery": filter_query, "pageSize": "1"},
        )
        first_page_items = response.get("items", [])
        assert len(first_page_items) == 1, f"Expected 1 item on first page, got {len(first_page_items)}"
        next_page_token = response.get("nextPageToken")
        assert next_page_token, "Expected nextPageToken for second page"

        # Second page
        response = execute_get_command(
            url=base_url,
            headers=model_registry_rest_headers,
            params={"filterQuery": filter_query, "pageSize": "1", "nextPageToken": next_page_token},
        )
        second_page_items = response.get("items", [])
        assert len(second_page_items) == 1, f"Expected 1 item on second page, got {len(second_page_items)}"

        assert first_page_items[0]["name"] != second_page_items[0]["name"], (
            "Pagination returned the same server on both pages"
        )
