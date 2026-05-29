import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap

MCP_CATALOG_SOURCES_CM: str = "mcp-catalog-sources"


def exclude_default_mcp_servers(response: dict, default_mcp_servers: dict) -> list[dict]:
    """Return only non-default servers from an API response by excluding default server IDs."""
    default_server_ids = {server["name"] for server in default_mcp_servers.get("items", [])}
    return [server for server in response.get("items", []) if server["name"] not in default_server_ids]


def get_mcp_catalog_sources(admin_client: DynamicClient, model_registry_namespace: str) -> tuple[ConfigMap, dict]:
    """Return the catalog ConfigMap and its parsed sources.yaml data."""
    catalog_config_map = ConfigMap(
        name=MCP_CATALOG_SOURCES_CM,
        client=admin_client,
        namespace=model_registry_namespace,
    )
    current_data = yaml.safe_load(catalog_config_map.instance.data.get("sources.yaml", "{}") or "{}")
    return catalog_config_map, current_data
