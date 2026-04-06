from typing import Self

import pytest
import structlog

from tests.model_registry.utils import execute_get_command

LOGGER = structlog.get_logger(name=__name__)


class TestMCPServerOrdering:
    """Tests for MCP server ordering functionality."""

    @pytest.mark.parametrize(
        "sort_order",
        [
            pytest.param("ASC", id="ascending"),
            pytest.param("DESC", id="descending"),
        ],
    )
    def test_ordering_by_name(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        sort_order: str,
    ):
        """TC-API-014: Test ordering MCP servers by name ASC/DESC."""
        response = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
            headers=model_registry_rest_headers,
            params={"orderBy": "name", "sortOrder": sort_order},
        )
        actual_names = [s["name"] for s in response.get("items", [])]
        expected_names = sorted(actual_names, reverse=(sort_order == "DESC"))
        assert actual_names == expected_names
