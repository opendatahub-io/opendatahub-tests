from typing import Any, Self

import pytest
import structlog

from tests.ai_hub.mcp_servers.preview.utils import validate_mcp_preview_counts, validate_preview_items

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.tier2
@pytest.mark.usefixtures("model_registry_namespace")
class TestMCPPreviewExistingSource:
    """
    Test class for validating the MCP server preview API using path mode against the default Red Hat MCP catalog.
    """

    @pytest.mark.parametrize(
        "mcp_preview_result",
        [
            pytest.param(
                {"included_servers": ["*"], "excluded_servers": ["*aap*"]},
                id="test_include_exclude",
            ),
            pytest.param(
                {},
                id="test_no_filters",
            ),
        ],
        indirect=True,
    )
    def test_mcp_preview_server_filters(
        self: Self,
        mcp_preview_result: tuple[dict[str, Any], list[str] | None, list[str] | None],
        default_mcp_servers_from_yaml: list[dict[str, Any]],
    ):
        """
        Test the catalog preview API for MCP servers with various filter combinations.

        Given an MCP server catalog with known server names,
        When previewing with or without includedServers/excludedServers glob patterns,
        Then the preview response should return accurate counts and correct included status per server.
        """
        result, included_patterns, excluded_patterns = mcp_preview_result
        catalog_server_names = {server["name"] for server in default_mcp_servers_from_yaml}
        validate_mcp_preview_counts(
            result=result,
            mcp_servers=default_mcp_servers_from_yaml,
            included_patterns=included_patterns,
            excluded_patterns=excluded_patterns,
        )
        validate_preview_items(
            result=result,
            included_patterns=included_patterns,
            excluded_patterns=excluded_patterns,
            expected_server_names=catalog_server_names,
        )

    @pytest.mark.parametrize(
        "mcp_preview_result, filter_status",
        [
            pytest.param(
                {"included_servers": ["*"], "excluded_servers": ["*aap*"], "filter_status": "all"},
                "all",
                id="test_filter_all",
            ),
            pytest.param(
                {"included_servers": ["*open*"], "filter_status": "included"},
                "included",
                id="test_filter_included",
            ),
            pytest.param(
                {"included_servers": ["*"], "excluded_servers": ["*aap*"], "filter_status": "excluded"},
                "excluded",
                id="test_filter_excluded",
            ),
        ],
        indirect=["mcp_preview_result"],
    )
    def test_mcp_preview_filter_status(
        self: Self,
        mcp_preview_result: tuple[dict[str, Any], list[str] | None, list[str] | None],
        filter_status: str,
    ):
        """
        Test the MCP preview API with filterStatus parameter.

        Given an MCP server catalog with include/exclude filters that produce both included and excluded servers,
        When previewing with filterStatus set to all, included, or excluded,
        Then only items matching the requested status should be returned.
        """
        result, included_patterns, excluded_patterns = mcp_preview_result

        validate_preview_items(
            result=result,
            included_patterns=included_patterns,
            excluded_patterns=excluded_patterns,
        )

        summary = result.get("summary")
        assert summary is not None, "Missing 'summary' in response"
        items = result.get("items", [])

        expected_key = {"all": "totalAssets", "included": "includedAssets", "excluded": "excludedAssets"}[filter_status]
        assert len(items) == summary[expected_key], (
            f"Items ({len(items)}) doesn't match {expected_key} ({summary[expected_key]})"
        )

        if filter_status in ("included", "excluded"):
            expected_value = filter_status == "included"
            wrong = [item["name"] for item in items if item.get("included") != expected_value]
            assert not wrong, f"filterStatus={filter_status} returned items with included={not expected_value}: {wrong}"
            server_names = [item["name"] for item in items]
            LOGGER.info(f"filterStatus={filter_status}: {server_names} all have included={expected_value}")
