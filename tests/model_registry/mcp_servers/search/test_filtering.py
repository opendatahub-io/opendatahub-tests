from typing import Self

import pytest
from simple_logger.logger import get_logger

from tests.model_registry.mcp_servers.constants import CALCULATOR_PROVIDER, CALCULATOR_SERVER_NAME
from tests.model_registry.utils import execute_get_command

LOGGER = get_logger(name=__name__)


@pytest.mark.usefixtures("mcp_servers_configmap_patch")
class TestMCPServerFiltering:
    """Tests for MCP server filterQuery functionality."""

    @pytest.mark.parametrize(
        "filter_query, expected_count, expected_name, field_check",
        [
            pytest.param(
                f"provider='{CALCULATOR_PROVIDER}'",
                1,
                CALCULATOR_SERVER_NAME,
                ("provider", CALCULATOR_PROVIDER),
                id="by_provider",
            ),
            pytest.param("tags='math'", 1, CALCULATOR_SERVER_NAME, None, id="by_tags"),
            pytest.param("license='BSD 3-Clause'", 1, "file-manager", ("license", "BSD 3-Clause"), id="by_license"),
        ],
    )
    def test_filter_by_field(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        filter_query: str,
        expected_count: int,
        expected_name: str,
        field_check: tuple[str, str] | None,
    ):
        """TC-API-003, TC-API-005, TC-API-009: Test filtering MCP servers by provider, tags, and license."""
        response = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
            headers=model_registry_rest_headers,
            params={"filterQuery": filter_query},
        )
        items = response.get("items", [])
        assert len(items) == expected_count, (
            f"Expected {expected_count} server(s) for '{filter_query}', got {len(items)}"
        )
        assert items[0]["name"] == expected_name
        if field_check:
            assert items[0][field_check[0]] == field_check[1]

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

        filters = response["filters"]

        expected_filter_fields = {"description", "provider", "license", "tags"}
        actual_filter_fields = set(filters.keys())
        assert expected_filter_fields == actual_filter_fields

    def test_pagination_with_filters(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """TC-API-032: Test that pagination works correctly with filterQuery."""
        base_url = f"{mcp_catalog_rest_urls[0]}mcp_servers"
        filter_query = "license='MIT'"

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

        collected_names = {first_page_items[0]["name"], second_page_items[0]["name"]}
        assert collected_names == {"weather-api", CALCULATOR_SERVER_NAME}
