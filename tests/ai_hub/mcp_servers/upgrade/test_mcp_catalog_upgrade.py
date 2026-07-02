from typing import Self

import pytest
import structlog
import yaml
from ocp_resources.config_map import ConfigMap

from tests.ai_hub.mcp_servers.config.constants import MCP_CATALOG_SOURCE_ID

LOGGER = structlog.get_logger(name=__name__)

pytestmark = [
    pytest.mark.usefixtures(
        "updated_dsc_component_state_scope_session",
        "model_registry_namespace",
    )
]


class TestPreUpgradeMCPCatalog:
    """Validates MCP catalog custom source is configured before upgrade."""

    @pytest.mark.order("first")
    @pytest.mark.pre_upgrade
    def test_validate_mcp_sources(
        self: Self,
        pre_upgrade_mcp_config_map_update: ConfigMap,
    ):
        """Given a custom MCP catalog source configured in the ConfigMap
        When reading the ConfigMap before upgrade
        Then the custom source is present in mcp_catalogs
        """
        sources_data = yaml.safe_load(pre_upgrade_mcp_config_map_update.instance.data["sources.yaml"])
        mcp_catalogs = sources_data.get("mcp_catalogs", [])
        matching = [cat for cat in mcp_catalogs if cat.get("id") == MCP_CATALOG_SOURCE_ID]
        assert matching, f"Custom MCP source '{MCP_CATALOG_SOURCE_ID}' not found in ConfigMap mcp_catalogs"
        LOGGER.info(f"Pre-upgrade: MCP catalog source '{MCP_CATALOG_SOURCE_ID}' is configured")
