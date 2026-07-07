from typing import Any, Self

import pytest

from tests.ai_hub.mcp_servers.preview.constants import MCP_SERVER_NAMES, MCP_SERVERS_LIST
from tests.ai_hub.mcp_servers.preview.utils import validate_mcp_preview_counts, validate_preview_items


@pytest.mark.tier2
@pytest.mark.usefixtures("model_registry_namespace")
class TestMCPPreviewStatelessMode:
    """
    Test class for validating the MCP server preview API with user-provided catalog data (stateless mode).
    """

    @pytest.mark.parametrize(
        "stateless_preview_result",
        [
            pytest.param(
                {"included_servers": ["kubernetes*"], "excluded_servers": ["*-experimental"]},
                id="test_include_and_exclude",
            ),
            pytest.param(
                {"included_servers": ["*-mcp"]},
                id="test_include_only",
            ),
            pytest.param(
                {"excluded_servers": ["grafana-*"]},
                id="test_exclude_only",
            ),
            pytest.param(
                {},
                id="test_no_filters",
            ),
        ],
        indirect=True,
    )
    def test_mcp_preview_stateless_counts(
        self: Self,
        stateless_preview_result: tuple[dict[str, Any], list[str] | None, list[str] | None],
    ):
        """
        Test that stateless MCP preview returns accurate summary counts for various filter combinations.

        Given custom MCP server data uploaded inline via the catalogData field,
        When previewing with different includedServers/excludedServers filter combinations,
        Then the summary counts should match the expected filtered results.
        """
        result, included_patterns, excluded_patterns = stateless_preview_result
        validate_mcp_preview_counts(
            result=result,
            mcp_servers=MCP_SERVERS_LIST,
            included_patterns=included_patterns,
            excluded_patterns=excluded_patterns,
        )

    @pytest.mark.parametrize(
        "stateless_preview_result",
        [
            pytest.param(
                {"included_servers": ["kubernetes*"], "excluded_servers": ["*-experimental"]},
                id="test_include_and_exclude",
            ),
        ],
        indirect=True,
    )
    def test_mcp_preview_stateless_items(
        self: Self,
        stateless_preview_result: tuple[dict[str, Any], list[str] | None, list[str] | None],
    ):
        """
        Test that stateless MCP preview returns correct per-item included status and valid server names.

        Given custom MCP server data uploaded inline via the catalogData field,
        When previewing with includedServers and excludedServers filters,
        Then each item should have the correct included flag and name should match the inline data.
        """
        result, included_patterns, excluded_patterns = stateless_preview_result
        validate_preview_items(
            result=result,
            included_patterns=included_patterns,
            excluded_patterns=excluded_patterns,
            expected_server_names=MCP_SERVER_NAMES,
        )
