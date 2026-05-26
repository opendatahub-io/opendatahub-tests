from typing import Self

import pytest
import structlog

from tests.model_registry.utils import execute_get_command

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.tier2
@pytest.mark.parametrize(
    "mcp_included_excluded_configmap_patch",
    [
        pytest.param(
            {"includedServers": ["weather-*", "file-*"], "excludedServers": ["file-*"]},
            id="combined_include_exclude",
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("mcp_included_excluded_configmap_patch")
class TestMCPServerIncludedExcludedFiltering:
    """Tests for includedServers/excludedServers glob pattern filtering (TC-LOAD-003, TC-LOAD-004, TC-LOAD-005)."""

    def test_included_and_excluded_servers(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Verify includedServers are loaded and excludedServers are not (TC-LOAD-003, TC-LOAD-004, TC-LOAD-005)."""
        response = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
            headers=model_registry_rest_headers,
            params={"pageSize": 1000},
        )
        server_names = {server["name"] for server in response.get("items", [])}
        assert "weather-api" in server_names, f"Expected 'weather-api' in {server_names}"
        assert "file-manager" not in server_names, f"Expected 'file-manager' not in {server_names}"
