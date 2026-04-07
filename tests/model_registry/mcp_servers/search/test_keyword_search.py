from typing import Self

import pytest
import structlog

from tests.model_registry.utils import execute_get_command

LOGGER = structlog.get_logger(name=__name__)


class TestMCPServerKeywordSearch:
    """Tests for MCP server keyword search via q parameter combined with other features."""

    @pytest.mark.parametrize(
        "params, expected_name",
        [
            pytest.param(
                {"q": "OpenShift", "filterQuery": "license='Apache 2.0'"},
                "openshift-mcp-server",
                id="with_filter_query",
            ),
            pytest.param(
                {"q": "Ansible", "filterQuery": "provider='Red Hat'"},
                "aap-mcp-server",
                id="with_filter_and_provider",
            ),
        ],
    )
    def test_keyword_search_combined(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        params: dict[str, str],
        expected_name: str,
    ):
        """TC-API-012: Test q parameter combined with filterQuery (AND logic)."""
        response = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
            headers=model_registry_rest_headers,
            params=params,
        )
        items = response.get("items", [])
        actual_names = {server["name"] for server in items}
        assert expected_name in actual_names, f"Expected '{expected_name}' in search results, got {actual_names}"
