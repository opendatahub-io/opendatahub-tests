import yaml
from ocp_resources.config_map import ConfigMap


def get_default_model_catalog_yaml(catalog_config_map: ConfigMap):
    """
    Extract catalogs from config map sources.yaml data.

    Args:
        catalog_config_map: The ConfigMap object containing the sources.yaml data

    Returns:
        The catalogs section from the sources.yaml data
    """
    return yaml.safe_load(catalog_config_map.instance.data["sources.yaml"])["catalogs"]
