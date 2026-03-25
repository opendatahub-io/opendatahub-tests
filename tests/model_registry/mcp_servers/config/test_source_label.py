from typing import Self

import pytest
import structlog

from tests.model_registry.mcp_servers.constants import (
    MCP_CATALOG_SOURCE2_NAME,
    MCP_CATALOG_SOURCE_NAME,
)
from tests.model_registry.utils import execute_get_command

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures("mcp_source_label_configmap_patch")
class TestMCPServerSourceLabel:
    """Tests for MCP server sourceLabel filtering (TC-API-036 to TC-API-039)."""

    @pytest.mark.smoke
    def test_mcp_server_source_label(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        Validate MCP server filtering by source label (TC-API-036, TC-API-038, TC-API-039).
        """
        source1_size = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
            headers=model_registry_rest_headers,
            params={"sourceLabel": MCP_CATALOG_SOURCE_NAME},
        )["size"]
        source2_size = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
            headers=model_registry_rest_headers,
            params={"sourceLabel": MCP_CATALOG_SOURCE2_NAME},
        )["size"]
        null_label_size = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
            headers=model_registry_rest_headers,
            params={"sourceLabel": "null"},
        )["size"]
        no_filtered_size = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
            headers=model_registry_rest_headers,
        )["size"]
        both_labeled_size = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
            headers=model_registry_rest_headers,
            params={"sourceLabel": f"{MCP_CATALOG_SOURCE_NAME},{MCP_CATALOG_SOURCE2_NAME}"},
        )["size"]
        LOGGER.info(f"no_filtered_size: {no_filtered_size}")
        assert no_filtered_size > 0
        assert null_label_size >= 0
        assert source1_size + source2_size == both_labeled_size
        assert no_filtered_size == both_labeled_size + null_label_size

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
