import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap

from tests.ai_hub.constants import DEFAULT_CUSTOM_MODEL_CATALOG

MCP_CATALOG_SOURCES_CM: str = "mcp-catalog-sources"
CATALOG_SOURCES_CONFIGMAP_NAMES: tuple[str, ...] = (
    DEFAULT_CUSTOM_MODEL_CATALOG,
    MCP_CATALOG_SOURCES_CM,
)


def exclude_default_mcp_servers(response: dict, default_mcp_servers: dict) -> list[dict]:
    """Return only non-default servers from an API response by excluding default server IDs."""
    default_server_ids = {server["name"] for server in default_mcp_servers.get("items", [])}
    return [server for server in response.get("items", []) if server["name"] not in default_server_ids]


def get_catalog_sources_configmap(admin_client: DynamicClient, model_registry_namespace: str) -> tuple[ConfigMap, dict]:
    """Return the active catalog sources ConfigMap and its parsed sources.yaml data.

    Tries model-catalog-sources first (current runtime ConfigMap), then
    mcp-catalog-sources for backward compatibility.
    """
    for configmap_name in CATALOG_SOURCES_CONFIGMAP_NAMES:
        catalog_config_map = ConfigMap(
            name=configmap_name,
            client=admin_client,
            namespace=model_registry_namespace,
        )
        if catalog_config_map.exists:
            current_data = yaml.safe_load(catalog_config_map.instance.data.get("sources.yaml", "{}") or "{}")
            return catalog_config_map, current_data
    raise AssertionError(
        f"No catalog sources ConfigMap found in {model_registry_namespace}; "
        f"tried: {', '.join(CATALOG_SOURCES_CONFIGMAP_NAMES)}"
    )


def get_mcp_catalog_sources(admin_client: DynamicClient, model_registry_namespace: str) -> tuple[ConfigMap, dict]:
    """Return the catalog ConfigMap and its parsed sources.yaml data."""
    return get_catalog_sources_configmap(
        admin_client=admin_client,
        model_registry_namespace=model_registry_namespace,
    )
