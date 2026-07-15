from typing import Any, Self

import pytest
import structlog

from tests.ai_hub.mcp_servers.config.constants import DEFAULT_MCP_LABEL, PARTNER_MCP_LABEL
from tests.ai_hub.utils import execute_get_command_with_retry

LOGGER = structlog.get_logger(name=__name__)

SUPPORT_TIER_RED_HAT: str = "redHatSupported"
SUPPORT_TIER_PARTNER: str = "partnerSupported"
SUPPORT_TIER_COMMUNITY: str = "communitySupported"
SUPPORT_TIER_FIELD: str = "supportTier"

pytestmark = [
    pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace"),
]


@pytest.mark.tier1
@pytest.mark.install
@pytest.mark.pre_upgrade
@pytest.mark.post_upgrade
@pytest.mark.parametrize(
    "mcp_servers_by_source, expected_tier",
    [
        pytest.param(DEFAULT_MCP_LABEL, SUPPORT_TIER_RED_HAT, id="test_redhat"),
        pytest.param(PARTNER_MCP_LABEL, SUPPORT_TIER_PARTNER, id="test_partner"),
        pytest.param("null", SUPPORT_TIER_COMMUNITY, id="test_community"),
    ],
    indirect=["mcp_servers_by_source"],
)
class TestSupportTierValues:
    """Tests that every MCP server carries the correct supportTier for its source.

    RHOAIENG-71270: The catalog service derives supportTier from file provenance.
    Red Hat-sourced servers get redHatSupported, partner-sourced get partnerSupported,
    and community-sourced get communitySupported.
    """

    def test_servers_have_correct_support_tier(
        self: Self,
        mcp_servers_by_source: dict[str, Any],
        expected_tier: str,
    ) -> None:
        """Verify that every MCP server from a given source has the expected supportTier.

        Given a set of MCP servers filtered by source label,
        When each server's supportTier field is inspected,
        Then every server must carry the expected tier value with no exceptions.
        """
        items = mcp_servers_by_source.get("items", [])
        assert items, f"Expected at least one server for tier '{expected_tier}', but got none"

        errors: list[str] = []
        for server in items:
            server_name = server.get("name", "<unknown>")
            actual_tier = server.get("customProperties", {}).get(SUPPORT_TIER_FIELD, {}).get("string_value")
            if actual_tier != expected_tier:
                errors.append(f"Server '{server_name}': expected supportTier='{expected_tier}', got '{actual_tier}'")

        assert not errors, f"supportTier validation failed for {len(errors)} server(s):\n" + "\n".join(errors)


class TestSupportTierFiltering:
    """Tests for filtering the MCP catalog by the supportTier field.

    RHOAIENG-71270: The /api/mcp_catalog/v1alpha1/mcp_servers endpoint must
    accept filterQuery=supportTier='<value>' and return only matching servers.
    """

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        "support_tier",
        [
            pytest.param(SUPPORT_TIER_RED_HAT, id="test_red_hat_supported"),
            pytest.param(SUPPORT_TIER_PARTNER, id="test_partner_supported"),
            pytest.param(SUPPORT_TIER_COMMUNITY, id="test_community_supported"),
        ],
    )
    def test_filter_by_support_tier(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        default_mcp_servers: dict[str, Any],
        support_tier: str,
    ) -> None:
        """Verify that filterQuery=supportTier='<value>' returns only matching servers.

        Given a baseline of all default MCP servers and a specific support tier,
        When a GET request is issued with filterQuery=supportTier='<tier>',
        Then the response contains exactly the servers that belong to that tier and no others.
        """
        filter_query = f"{SUPPORT_TIER_FIELD}='{support_tier}'"
        LOGGER.info(f"Filtering MCP servers by: {filter_query}")

        response = execute_get_command_with_retry(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
            headers=model_registry_rest_headers,
            params={"filterQuery": filter_query, "pageSize": 1000},
        )
        items = response.get("items", [])

        expected_names = {
            server["name"]
            for server in default_mcp_servers.get("items", [])
            if server.get("customProperties", {}).get(SUPPORT_TIER_FIELD, {}).get("string_value") == support_tier
        }
        assert expected_names, (
            f"No default servers found with {SUPPORT_TIER_FIELD}='{support_tier}' in baseline; "
            "cannot validate filter results"
        )
        assert len(items) == len(expected_names), (
            f"Filter '{filter_query}' returned {len(items)} server(s), "
            f"expected at least {len(expected_names)} (known defaults: {expected_names})"
        )

        returned_names = {item["name"] for item in items}
        assert expected_names == returned_names, (
            f"Known default servers {expected_names} missing from filtered results {returned_names}"
        )

        errors: list[str] = []
        for item in items:
            actual_tier = item.get("customProperties", {}).get(SUPPORT_TIER_FIELD, {}).get("string_value")
            if actual_tier != support_tier:
                errors.append(
                    f"Server '{item.get('name', '<unknown>')}' returned with "
                    f"{SUPPORT_TIER_FIELD}='{actual_tier}', expected '{support_tier}'"
                )
        assert not errors, f"Filter '{filter_query}' returned servers with wrong tier:\n" + "\n".join(errors)

    @pytest.mark.tier3
    def test_filter_by_nonexistent_support_tier(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
    ) -> None:
        """Verify that filtering by an unknown supportTier returns an empty result set.

        Given that no MCP server has supportTier='nonExistent',
        When a GET request is issued with filterQuery=supportTier='nonExistent',
        Then the response contains zero items.
        """
        filter_query = f"{SUPPORT_TIER_FIELD}='nonExistent'"
        LOGGER.info(f"Filtering MCP servers by nonexistent tier: {filter_query}")

        response = execute_get_command_with_retry(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
            headers=model_registry_rest_headers,
            params={"filterQuery": filter_query},
        )
        size = response.get("size", -1)
        assert size == 0, f"Expected 0 servers for filter '{filter_query}', but got size={size}"
