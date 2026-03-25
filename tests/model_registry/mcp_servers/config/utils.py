import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap

from tests.model_registry.constants import DEFAULT_CUSTOM_MODEL_CATALOG


def get_mcp_catalog_sources(admin_client: DynamicClient, model_registry_namespace: str) -> tuple[ConfigMap, dict]:
    """Return the catalog ConfigMap and its parsed sources.yaml data."""
    catalog_config_map = ConfigMap(
        name=DEFAULT_CUSTOM_MODEL_CATALOG,
        client=admin_client,
        namespace=model_registry_namespace,
    )
    current_data = yaml.safe_load(catalog_config_map.instance.data.get("sources.yaml", "{}") or "{}")
    return catalog_config_map, current_data
