from typing import Any, Self

import pytest
import structlog

from tests.model_registry.model_catalog.constants import (
    OTHER_MODELS_CATALOG_ID,
    REDHAT_AI_CATALOG_ID,
    VALIDATED_CATALOG_ID,
)
from tests.model_registry.utils import execute_get_command

pytestmark = [pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace")]

DEFAULT_MCP_CATALOG_ID: str = "rh_mcp_servers"
LOGGER = structlog.get_logger(name=__name__)


class TestSourcesEndpoint:
    """Test class for the model catalog sources endpoint."""

    @pytest.mark.parametrize(
        "sparse_override_catalog_source",
        [{"id": REDHAT_AI_CATALOG_ID, "field_name": "enabled", "field_value": False}],
        indirect=True,
    )
    @pytest.mark.tier1
    def test_sources_endpoint_returns_all_sources_regardless_of_enabled_field(
        self,
        sparse_override_catalog_source: dict,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        Test that sources endpoint returns ALL sources regardless of enabled field value.
        """
        response = execute_get_command(url=f"{model_catalog_rest_url[0]}sources", headers=model_registry_rest_headers)
        items = response.get("items", [])

        assert len(items) > 1, "Expected multiple sources to be returned"

        # Verify we have at least one enabled source
        enabled_sources = [item for item in items if item.get("status") == "available"]
        assert enabled_sources, "Expected at least one enabled source"

        # Verify we have at least one disabled source
        disabled_sources = [item for item in items if item.get("status") == "disabled"]
        assert disabled_sources, "Expected at least one disabled source"

        assert len(enabled_sources) + len(disabled_sources) == len(items), "Expected all sources to be returned"

        LOGGER.info(
            f"Sources endpoint returned {len(items)} total sources: "
            f"{len(enabled_sources)} enabled, {len(disabled_sources)} disabled"
        )


class TestAssetTypeFilter:
    """Tests for /sources endpoint assetType query parameter filtering."""

    @pytest.mark.parametrize(
        "asset_type,expected_ids",
        [
            (None, {REDHAT_AI_CATALOG_ID, VALIDATED_CATALOG_ID, OTHER_MODELS_CATALOG_ID}),
            ("models", {REDHAT_AI_CATALOG_ID, VALIDATED_CATALOG_ID, OTHER_MODELS_CATALOG_ID}),
            ("mcp_servers", {DEFAULT_MCP_CATALOG_ID}),
            ("invalid_value", set()),
        ],
        ids=["default-models", "explicit-models", "mcp-servers", "invalid-empty"],
    )
    def test_asset_type_filters_sources(
        self: Self,
        asset_type: str | None,
        expected_ids: set[str],
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ) -> None:
        """Test that the /sources endpoint filters by assetType, returning only catalog sources
        for None/models, only MCP sources for mcp_servers, and an empty list for invalid values.
        """
        sources_url = f"{model_catalog_rest_url[0]}sources"
        params: dict[str, Any] = {}
        if asset_type is not None:
            params["assetType"] = asset_type

        response = execute_get_command(url=sources_url, headers=model_registry_rest_headers, params=params or None)
        source_ids = {item["id"] for item in response["items"]}

        assert source_ids == expected_ids
        LOGGER.info(f"assetType={asset_type} returned sources: {source_ids}")
