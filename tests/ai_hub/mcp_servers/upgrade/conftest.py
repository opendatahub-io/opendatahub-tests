import pytest
import structlog
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.resource import ResourceEditor

from tests.ai_hub.mcp_servers.config.constants import (
    MCP_CATALOG_SOURCE,
    MCP_SERVERS_YAML,
)
from tests.ai_hub.mcp_servers.config.utils import get_mcp_catalog_sources
from tests.ai_hub.utils import wait_for_mcp_catalog_api, wait_for_model_catalog_pod_ready_after_deletion

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="class")
def pre_upgrade_mcp_config_map_update(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    mcp_catalog_rest_urls: list[str],
    model_registry_rest_headers: dict[str, str],
) -> ConfigMap:
    """Patches MCP catalog ConfigMap with a custom source before upgrade."""
    catalog_config_map, current_data = get_mcp_catalog_sources(
        admin_client=admin_client, model_registry_namespace=model_registry_namespace
    )
    if "mcp_catalogs" not in current_data:
        current_data["mcp_catalogs"] = []
    current_data["mcp_catalogs"].append(MCP_CATALOG_SOURCE)

    patches = {
        "data": {
            "sources.yaml": yaml.dump(current_data, default_flow_style=False),
            "mcp-servers.yaml": MCP_SERVERS_YAML,
        }
    }

    ResourceEditor(patches={catalog_config_map: patches}).update()
    wait_for_model_catalog_pod_ready_after_deletion(
        client=admin_client, model_registry_namespace=model_registry_namespace
    )
    wait_for_mcp_catalog_api(url=mcp_catalog_rest_urls[0], headers=model_registry_rest_headers)
    return catalog_config_map
