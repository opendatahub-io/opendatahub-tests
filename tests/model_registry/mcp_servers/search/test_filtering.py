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

        expected_filters = {
            "transports",
            "license",
            "provider",
            "architecture.array_value",
            "publishedDate",
            "deploymentMode",
            "tags",
        }
        assert expected_filters == set(response["filters"].keys())

    @pytest.mark.parametrize(
        "order_params",
        [
            pytest.param({}, id="test_without_order_by"),
            pytest.param({"orderBy": "id", "sortOrder": "ASC"}, id="test_with_order_by_asc"),
            pytest.param({"orderBy": "id", "sortOrder": "DESC"}, id="test_with_order_by_desc"),
        ],
    )
    def test_pagination_with_filters(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        order_params: dict[str, str],
    ):
        """TC-API-032: Test that pagination works correctly with filterQuery.

        Paginates through all filtered results one item at a time and verifies
        that every page returns a unique, previously unseen server ID.
        """
        base_url = f"{mcp_catalog_rest_urls[0]}mcp_servers"
        expected_license = "Apache 2.0"
        filter_query = f"license='{expected_license}'"

        # Get total count of matching servers
        all_response = execute_get_command(
            url=base_url,
            headers=model_registry_rest_headers,
            params={"filterQuery": filter_query},
        )
        total_items = all_response.get("size", 0)
        LOGGER.info(f"Total items matching filter '{filter_query}': {total_items}")
        assert total_items >= 2, f"Need at least 2 servers matching filter for pagination test, got {total_items}"

        seen_ids: list[str] = []
        next_page_token = None

        for page_num in range(1, total_items + 1):
            params: dict[str, str] = {"filterQuery": filter_query, "pageSize": "1", **order_params}
            if next_page_token:
                params["nextPageToken"] = next_page_token

            response = execute_get_command(
                url=base_url,
                headers=model_registry_rest_headers,
                params=params,
            )
            items = response.get("items", [])

            assert len(items) == 1, f"Expected 1 item on page {page_num}, got {len(items)}"

            assert items[0]["license"] == expected_license, (
                f"Page {page_num} server id '{items[0]['id']}' has license='{items[0].get('license')}',"
                f" expected '{expected_license}'"
            )

            server_id = items[0]["id"]
            assert server_id not in seen_ids, (
                f"Page {page_num} returned duplicate server id '{server_id}', already seen on a previous page"
            )
            sort_order = order_params.get("sortOrder", "ASC")
            if seen_ids:
                if sort_order == "ASC":
                    assert server_id > seen_ids[-1], (
                        f"Page {page_num} id '{server_id}' is not greater than previous id '{seen_ids[-1]}'"
                    )
                elif sort_order == "DESC":
                    assert server_id < seen_ids[-1], (
                        f"Page {page_num} id '{server_id}' is not less than previous id '{seen_ids[-1]}'"
                    )
            seen_ids.append(server_id)

            next_page_token = response.get("nextPageToken")
            if page_num < total_items:
                assert next_page_token, f"Expected nextPageToken after page {page_num}, but got none"

        LOGGER.info(f"Pagination complete: found {len(seen_ids)} unique servers out of {total_items} total")
        assert len(seen_ids) == total_items, (
            f"Pagination returned {len(seen_ids)} unique servers but expected {total_items}"
        )
