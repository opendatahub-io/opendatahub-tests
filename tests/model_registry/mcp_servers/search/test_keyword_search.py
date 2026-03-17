from typing import Self

import pytest
from simple_logger.logger import get_logger

from tests.model_registry.mcp_servers.constants import CALCULATOR_SERVER_NAME
from tests.model_registry.utils import execute_get_command

LOGGER = get_logger(name=__name__)


@pytest.mark.usefixtures("mcp_servers_configmap_patch")
class TestMCPServerKeywordSearch:
    """Tests for MCP server keyword search via q parameter combined with other features."""

    @pytest.mark.parametrize(
        "params, expected_names",
        [
            pytest.param(
                {"q": "Community", "filterQuery": "license='MIT'"},
                {"weather-api", CALCULATOR_SERVER_NAME},
                id="with_filter_query",
            ),
            pytest.param(
                {"q": "Math", "namedQuery": "production_ready"},
                {CALCULATOR_SERVER_NAME},
                id="with_named_query",
            ),
        ],
    )
    def test_keyword_search_combined(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        params: dict[str, str],
        expected_names: set[str],
    ):
        """TC-API-012: Test q parameter combined with filterQuery or namedQuery (AND logic)."""
        response = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
            headers=model_registry_rest_headers,
            params=params,
        )
        items = response.get("items", [])
        actual_names = {server["name"] for server in items}
        assert actual_names == expected_names
