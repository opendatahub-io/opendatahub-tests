import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap

from tests.model_registry.constants import DEFAULT_MCP_CATALOG_CM


def get_mcp_catalog_sources(admin_client: DynamicClient, model_registry_namespace: str) -> tuple[ConfigMap, dict]:
    """Return the MCP catalog ConfigMap and its parsed sources.yaml data."""
    catalog_config_map = ConfigMap(
        name=DEFAULT_MCP_CATALOG_CM,
        client=admin_client,
        namespace=model_registry_namespace,
    )
    current_data = yaml.safe_load(catalog_config_map.instance.data.get("sources.yaml", "{}") or "{}")
    return catalog_config_map, current_data
