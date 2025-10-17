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
from .constants import TEST_DATA

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="function")
def custom_catalog_setup(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    catalog_config_map: ConfigMap,
):
    """Fixture to setup custom catalog configuration for upgrade testing"""

    # Only run this setup during pre-upgrade tests
    if not request.config.getoption("--pre-upgrade", default=False):
        raise RuntimeError(
            "custom_catalog_setup fixture should only be used with --pre-upgrade option. "
            "This fixture creates permanent catalog modifications that are intended for upgrade testing."
        )

    LOGGER.info("Starting custom catalog setup via fixture")

    # NOTE: This catalog configuration setup is not applicable to 3.0 and will be updated.
    # The current approach of directly manipulating ConfigMaps may not work with
    # the new catalog architecture planned for version 3.0.

    # Get existing catalog configuration to preserve it
    existing_catalogs = get_default_model_catalog_yaml(config_map=catalog_config_map)

    # Generate custom catalog configuration
    custom_catalog_yaml = get_catalog_str(ids=[TEST_DATA["catalog_id"]])
    custom_sample_yaml_content = get_sample_yaml_str(models=TEST_DATA["models"])
    custom_sample_yaml_filename = f"{TEST_DATA['catalog_id'].replace('_', '-')}.yaml"

    # Parse custom catalog entries properly
    custom_catalog_entries = yaml.safe_load(f"catalogs:\n{custom_catalog_yaml}")["catalogs"]

    # Check if custom catalog already exists - this should be treated as a bug
    existing_catalog_ids = {catalog["id"] for catalog in existing_catalogs}
    for catalog_entry in custom_catalog_entries:
        if catalog_entry["id"] in existing_catalog_ids:
            raise RuntimeError(
                f"Custom catalog {catalog_entry['id']} already exists in the system."
                f"This indicates an unclean test environment or previous test cleanup failure."
                f"Existing catalog IDs: {existing_catalog_ids}"
            )

    # Combine catalogs (we know there are no duplicates at this point)
    combined_catalogs = existing_catalogs + custom_catalog_entries

    combined_sources_yaml = yaml.dump({"catalogs": combined_catalogs}, default_flow_style=False)

    # Store original configmap state for potential rollback
    original_configmap_data = (catalog_config_map.instance.data or {}).copy()

    # Prepare data for ResourceEditor pattern
    new_data = {"sources.yaml": combined_sources_yaml, custom_sample_yaml_filename: custom_sample_yaml_content}

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
