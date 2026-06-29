from typing import Any, Self

import pytest
import structlog

from tests.ai_hub.mcp_servers.preview.constants import MCP_SERVERS_LIST, TOTAL_SERVERS
from tests.ai_hub.utils import should_include_by_pattern

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.tier2
@pytest.mark.usefixtures("model_registry_namespace")
class TestMCPPreviewPagination:
    """
    Test class for validating the MCP server preview API pagination using stateless mode.
    """

    @pytest.mark.parametrize(
        "paginated_preview_pages",
        [
            pytest.param(
                {"page_size": 2, "included_servers": ["*"]},
                id="test_all_included_page_size_2",
            ),
            pytest.param(
                {"page_size": 2, "included_servers": ["kubernetes*"], "excluded_servers": ["*-experimental"]},
                id="test_filtered_page_size_2",
            ),
        ],
        indirect=True,
    )
    def test_mcp_preview_pages_return_all_servers(
        self: Self,
        paginated_preview_pages: list[dict[str, Any]],
    ):
        """
        Test that paginating through preview results returns all servers without duplicates.

        Given MCP servers uploaded inline,
        When fetching pages until nextPageToken is empty,
        Then all servers should be returned exactly once across all pages.
        """
        all_names: list[str] = []
        for page_number, page in enumerate(paginated_preview_pages, start=1):
            items = page.get("items", [])
            all_names.extend(item["name"] for item in items)
            LOGGER.info(f"Page {page_number}: {[item['name'] for item in items]}")

        LOGGER.info(f"All servers across {len(paginated_preview_pages)} pages: {all_names}")
        expected_total = paginated_preview_pages[0].get("summary", {}).get("totalAssets", TOTAL_SERVERS)
        assert len(all_names) == expected_total, f"Expected {expected_total} total items, got {len(all_names)}"
        assert len(set(all_names)) == len(all_names), f"Duplicate server names: {all_names}"

    @pytest.mark.parametrize(
        "paginated_preview_pages, included_servers, excluded_servers",
        [
            pytest.param(
                {"page_size": 2, "included_servers": ["*"]},
                ["*"],
                None,
                id="test_all_included",
            ),
            pytest.param(
                {"page_size": 2, "included_servers": ["kubernetes*"], "excluded_servers": ["*-experimental"]},
                ["kubernetes*"],
                ["*-experimental"],
                id="test_filtered",
            ),
        ],
        indirect=["paginated_preview_pages"],
    )
    def test_mcp_preview_summary_consistent_across_pages(
        self: Self,
        paginated_preview_pages: list[dict[str, Any]],
        included_servers: list[str] | None,
        excluded_servers: list[str] | None,
    ):
        """
        Test that summary counts remain consistent and correct across paginated requests.

        Given MCP servers uploaded inline with include/exclude filters,
        When fetching multiple pages,
        Then totalAssets/includedAssets/excludedAssets should be identical on every page and match expected counts.
        """
        first_summary = paginated_preview_pages[0].get("summary")
        assert first_summary is not None, "Missing 'summary' on first page"

        for page_number, page in enumerate(paginated_preview_pages[1:], start=2):
            summary = page.get("summary")
            assert summary == first_summary, f"Summary changed on page {page_number}: {summary} != {first_summary}"

        expected_included = sum(
            1
            for server in MCP_SERVERS_LIST
            if should_include_by_pattern(
                name=server["name"],
                included_patterns=included_servers,
                excluded_patterns=excluded_servers,
            )
        )
        expected_excluded = TOTAL_SERVERS - expected_included

        LOGGER.info(
            f"Summary validation: totalAssets={first_summary['totalAssets']}, "
            f"includedAssets={first_summary['includedAssets']} (expected {expected_included}), "
            f"excludedAssets={first_summary['excludedAssets']} (expected {expected_excluded})"
        )
        assert first_summary["totalAssets"] == TOTAL_SERVERS, (
            f"Expected totalAssets={TOTAL_SERVERS}, got {first_summary['totalAssets']}"
        )
        assert first_summary["includedAssets"] == expected_included, (
            f"Expected includedAssets={expected_included}, got {first_summary['includedAssets']}"
        )
        assert first_summary["excludedAssets"] == expected_excluded, (
            f"Expected excludedAssets={expected_excluded}, got {first_summary['excludedAssets']}"
        )
