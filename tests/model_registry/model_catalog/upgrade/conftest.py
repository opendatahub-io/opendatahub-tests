import pytest
import yaml
from kubernetes.dynamic import DynamicClient

from ocp_resources.config_map import ConfigMap
from ocp_resources.resource import ResourceEditor
from simple_logger.logger import get_logger

from tests.model_registry.model_catalog.utils import (
    is_model_catalog_ready,
    get_catalog_str,
    get_sample_yaml_str,
    get_default_model_catalog_yaml,
)
from tests.model_registry.constants import DEFAULT_MODEL_CATALOG
from .constants import TEST_DATA

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="function")
def custom_catalog_setup(admin_client: DynamicClient, model_registry_namespace: str):
    """Fixture to setup custom catalog configuration for upgrade testing"""
    LOGGER.info("Starting custom catalog setup via fixture")

    # Create Custom Catalog Source
    catalog_config_map = ConfigMap(name=DEFAULT_MODEL_CATALOG, client=admin_client, namespace=model_registry_namespace)

    # Get existing catalog configuration to preserve it
    existing_catalogs = get_default_model_catalog_yaml(config_map=catalog_config_map)

    # Generate custom catalog configuration
    custom_catalog_yaml = get_catalog_str(ids=[TEST_DATA["catalog_id"]])
    custom_sample_yaml_content = get_sample_yaml_str(models=TEST_DATA["models"])
    custom_sample_yaml_filename = f"{TEST_DATA['catalog_id'].replace('_', '-')}.yaml"

    # Parse custom catalog entries properly
    custom_catalog_entries = yaml.safe_load(f"catalogs:\n{custom_catalog_yaml}")["catalogs"]

    # Check if custom catalog already exists to avoid duplicates
    existing_catalog_ids = {catalog["id"] for catalog in existing_catalogs}
    new_catalog_entries = [catalog for catalog in custom_catalog_entries if catalog["id"] not in existing_catalog_ids]

    # Combine catalogs properly as Python objects (only add if not already present)
    if new_catalog_entries:
        combined_catalogs = existing_catalogs + new_catalog_entries
    else:
        combined_catalogs = existing_catalogs
        LOGGER.info(f"Custom catalog {TEST_DATA['catalog_id']} already exists, skipping addition")

    combined_sources_yaml = yaml.dump({"catalogs": combined_catalogs}, default_flow_style=False)

    # Store original configmap state for potential rollback
    original_configmap_data = (catalog_config_map.instance.data or {}).copy()

    # Prepare data for ResourceEditor pattern
    new_data = {"sources.yaml": combined_sources_yaml, custom_sample_yaml_filename: custom_sample_yaml_content}

    # Preserve any existing data that we're not modifying
    for key, value in original_configmap_data.items():
        if key not in new_data:
            new_data[key] = value

    # Apply permanent configuration changes using ResourceEditor.update() for persistent changes
    patches = {catalog_config_map: {"data": new_data}}

    ResourceEditor(patches=patches).update()

    # Wait for model catalog to pick up changes
    is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)

    LOGGER.info("Custom catalog setup completed via fixture")

    # Return useful information for tests
    return {
        "catalog_id": TEST_DATA["catalog_id"],
        "models": TEST_DATA["models"],
        "config_map": catalog_config_map,
        "original_data": original_configmap_data,
    }
