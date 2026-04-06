from typing import Self

import pytest
import structlog

from tests.model_registry.mcp_servers.config.constants import CALCULATOR_PROVIDER, CALCULATOR_SERVER_NAME
from tests.model_registry.mcp_servers.config.utils import exclude_default_mcp_servers
from tests.model_registry.utils import execute_get_command

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures("mcp_servers_configmap_patch")
class TestMCPServerNamedQueries:
    """Tests for MCP server named query functionality."""

    @pytest.mark.parametrize(
        "named_query, expected_custom_properties",
        [
            pytest.param(
                "production_ready",
                {"verifiedSource": True},
                id="production_ready",
            ),
            pytest.param(
                "security_focused",
                {"sast": True, "readOnlyTools": True},
                id="security_focused",
            ),
        ],
    )
    def test_named_query_execution(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        default_mcp_servers: dict,
        named_query: str,
        expected_custom_properties: dict[str, bool],
    ):
        """TC-API-011: Test executing a named query filters servers by custom properties."""
        response = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
            headers=model_registry_rest_headers,
            params={"namedQuery": named_query},
        )
        items = exclude_default_mcp_servers(response=response, default_mcp_servers=default_mcp_servers)
        assert len(items) == 1, f"Expected 1 server matching '{named_query}', got {len(items)}"
        assert items[0]["name"] == CALCULATOR_SERVER_NAME

        custom_props = items[0]["customProperties"]
        for prop_name, expected_value in expected_custom_properties.items():
            assert custom_props.get(prop_name, {}).get("bool_value") is expected_value, (
                f"Expected {prop_name}={expected_value}, got {custom_props.get(prop_name)}"
            )

    @pytest.mark.parametrize(
        "filter_query, expected_count, expected_names",
        [
            pytest.param(
                f"provider='{CALCULATOR_PROVIDER}'",
                1,
                {CALCULATOR_SERVER_NAME},
                id="matching_overlap",
            ),
            pytest.param(
                "provider='Weather Community'",
                0,
                set(),
                id="no_overlap",
            ),
        ],
    )
    def test_named_query_combined_with_filter_query(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        filter_query: str,
        expected_count: int,
        expected_names: set[str],
    ):
        """TC-API-013: Test combining namedQuery with filterQuery."""
        response = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
            headers=model_registry_rest_headers,
            params={"namedQuery": "production_ready", "filterQuery": filter_query},
        )
        items = response["items"]
        assert len(items) == expected_count, (
            f"Expected {expected_count} server(s) for namedQuery + '{filter_query}', got {len(items)}"
        )
        assert {server["name"] for server in items} == expected_names
