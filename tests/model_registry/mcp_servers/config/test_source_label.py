from typing import Self

import pytest
import structlog

from tests.model_registry.mcp_servers.config.constants import (
    DEFAULT_MCP_LABEL,
    MCP_CATALOG_SOURCE2_NAME,
    MCP_CATALOG_SOURCE_NAME,
)
from tests.model_registry.utils import execute_get_command

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures("mcp_source_label_configmap_patch")
class TestMCPServerSourceLabel:
    """Tests for MCP server sourceLabel filtering (TC-API-036 to TC-API-039)."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "source_label_param",
        [
            pytest.param({"sourceLabel": "null"}, id="null_label"),
            pytest.param({}, id="no_filter"),
            pytest.param({"sourceLabel": DEFAULT_MCP_LABEL}, id="default_label"),
        ],
    )
    def test_mcp_server_source_label(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        source_label_param: dict,
    ):
        """
        Validate MCP server filtering by source label (TC-API-036, TC-API-038, TC-API-039).
        """
        size = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
            headers=model_registry_rest_headers,
            params=source_label_param or None,
        )["size"]
        LOGGER.info(f"Source label filter {source_label_param}: size={size}")
        assert size > 0, f"Expected size > 0 for source label filter {source_label_param}, but got {size}"

    def test_mcp_server_custom_source_labels(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Validate MCP server filtering by individual and combined custom source labels."""
        sizes = {}
        for label in [
            MCP_CATALOG_SOURCE_NAME,
            MCP_CATALOG_SOURCE2_NAME,
            f"{MCP_CATALOG_SOURCE_NAME},{MCP_CATALOG_SOURCE2_NAME}",
        ]:
            size = execute_get_command(
                url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
                headers=model_registry_rest_headers,
                params={"sourceLabel": label},
            )["size"]
            LOGGER.info(f"Source label '{label}': size={size}")
            assert size > 0, f"Expected size > 0 for sourceLabel '{label}', but got {size}"
            sizes[label] = size

        combined_label = f"{MCP_CATALOG_SOURCE_NAME},{MCP_CATALOG_SOURCE2_NAME}"
        assert sizes[MCP_CATALOG_SOURCE_NAME] + sizes[MCP_CATALOG_SOURCE2_NAME] == sizes[combined_label], (
            f"Expected source1 ({sizes[MCP_CATALOG_SOURCE_NAME]}) + source2 ({sizes[MCP_CATALOG_SOURCE2_NAME]}) "
            f"== combined ({sizes[combined_label]})"
        )

    @pytest.mark.tier3
    def test_mcp_server_invalid_source_label(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        Validate MCP server filtering by invalid source label (TC-API-037).
        """
        invalid_size = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
            headers=model_registry_rest_headers,
            params={"sourceLabel": "invalid"},
        )["size"]

        assert invalid_size == 0
