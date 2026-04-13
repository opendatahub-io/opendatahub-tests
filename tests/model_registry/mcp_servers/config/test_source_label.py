from typing import Self

import pytest
import structlog

from tests.model_registry.mcp_servers.config.constants import (
    DEFAULT_MCP_LABEL,
    PARTNER_MCP_LABEL,
)
from tests.model_registry.utils import execute_get_command

LOGGER = structlog.get_logger(name=__name__)


class TestMCPServerSourceLabel:
    """Tests for MCP server sourceLabel filtering (TC-API-036 to TC-API-039)."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "source_label_param",
        [
            pytest.param({"sourceLabel": "null"}, id="test_null_label"),
            pytest.param({}, id="test_no_filter"),
            pytest.param({"sourceLabel": DEFAULT_MCP_LABEL}, id="test_default_label"),
            pytest.param({"sourceLabel": PARTNER_MCP_LABEL}, id="test_partner_label"),
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

    def test_mcp_server_source_label_combined(
        self: Self,
        mcp_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Validate MCP server filtering by individual and combined source labels."""
        default_label_size = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
            headers=model_registry_rest_headers,
            params={"sourceLabel": DEFAULT_MCP_LABEL},
        )["size"]
        partner_label_size = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
            headers=model_registry_rest_headers,
            params={"sourceLabel": PARTNER_MCP_LABEL},
        )["size"]
        both_labeled_size = execute_get_command(
            url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
            headers=model_registry_rest_headers,
            params={"sourceLabel": f"{DEFAULT_MCP_LABEL},{PARTNER_MCP_LABEL}"},
        )["size"]

        LOGGER.info(
            f"default_label_size: {default_label_size}, partner_label_size: {partner_label_size}, "
            f"both_labeled_size: {both_labeled_size}"
        )
        assert default_label_size > 0, f"Expected size > 0 for default label, but got {default_label_size}"
        assert partner_label_size > 0, f"Expected size > 0 for partner label, but got {partner_label_size}"
        assert default_label_size + partner_label_size == both_labeled_size, (
            f"Expected default ({default_label_size}) + partner ({partner_label_size}) "
            f"== combined ({both_labeled_size})"
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
