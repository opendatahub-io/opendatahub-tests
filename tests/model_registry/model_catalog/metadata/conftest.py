import pytest
import yaml
from typing import Generator

from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.resource import ResourceEditor

from tests.model_registry.constants import DEFAULT_CUSTOM_MODEL_CATALOG
from tests.model_registry.utils import is_model_catalog_ready, wait_for_model_catalog_api


@pytest.fixture()
def error_catalog_source(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> Generator[str, None, None]:
    """
    Creates a catalog source with an invalid configuration that will cause an error status.

    Yields:
        str: The catalog ID of the error catalog source.
    """
    sources_cm = ConfigMap(name=DEFAULT_CUSTOM_MODEL_CATALOG, client=admin_client, namespace=model_registry_namespace)
    current_data = yaml.safe_load(sources_cm.instance.data["sources.yaml"])

    error_catalog = {
        "id": "test-error-catalog",
        "name": "Test Error Catalog",
        "type": "yaml",
        "enabled": True,
        "properties": {"yamlCatalogPath": "/nonexistent/path/to/catalog.yaml"},
    }

    if "catalogs" not in current_data:
        current_data["catalogs"] = []
    current_data["catalogs"].append(error_catalog)

    patches = {"data": {"sources.yaml": yaml.dump(current_data, default_flow_style=False)}}

    with ResourceEditor(patches={sources_cm: patches}):
        is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
        wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

        yield error_catalog["id"]

    is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)


@pytest.fixture()
def disabled_catalog_source(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> Generator[str, None, None]:
    """
    Disables an existing catalog source for testing.

    Yields:
        str: The catalog ID of the disabled catalog.
    """
    sources_cm = ConfigMap(name=DEFAULT_CUSTOM_MODEL_CATALOG, client=admin_client, namespace=model_registry_namespace)
    current_data = yaml.safe_load(sources_cm.instance.data["sources.yaml"])

    # Get catalog_id from parameter (required)
    catalog_id = request.param
    assert catalog_id, "catalog_id parameter is required for disabled_catalog_source fixture"

    # Find catalog by ID
    catalog_to_disable = next(
        (catalog for catalog in current_data.get("catalogs", []) if catalog["id"] == catalog_id), None
    )
    assert catalog_to_disable is not None, f"Catalog '{catalog_id}' not found in ConfigMap"

    # Disable the catalog
    catalog_to_disable["enabled"] = False

    patches = {"data": {"sources.yaml": yaml.dump(current_data, default_flow_style=False)}}

    with ResourceEditor(patches={sources_cm: patches}):
        is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
        wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

        yield catalog_id

    is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
