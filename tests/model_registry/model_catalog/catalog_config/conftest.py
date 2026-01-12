import pytest
import yaml
from ocp_resources.config_map import ConfigMap
from ocp_resources.resource import ResourceEditor
from typing import Generator

from tests.model_registry.constants import DEFAULT_CUSTOM_MODEL_CATALOG
from tests.model_registry.utils import (
    execute_get_command,
    is_model_catalog_ready,
    wait_for_model_catalog_api,
)


@pytest.fixture()
def sparse_override_catalog_source(
    request: pytest.FixtureRequest,
    admin_client,
    model_registry_namespace: str,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> Generator[dict, None, None]:
    """
    Creates a sparse override for an existing default catalog source.

    Requires parameterization via request.param dict containing:
    - "id": catalog ID to override (required)
    - "field_name": name of the field to override (required)
    - "field_value": value for the field (required)
    """
    # Get fields from pytest param
    param = getattr(request, "param", None)
    assert param, "sparse_override_catalog_source requires request.param dict"

    catalog_id = param["id"]
    field_name = param["field_name"]
    field_value = param["field_value"]

    # Capture CURRENT catalog state from API before applying sparse override
    response = execute_get_command(url=f"{model_catalog_rest_url[0]}sources", headers=model_registry_rest_headers)
    items = response.get("items", [])
    original_catalog = next((item for item in items if item.get("id") == catalog_id), None)
    assert original_catalog is not None, f"Original catalog '{catalog_id}' not found in sources"

    # Create sparse override YAML with only id and the field to override
    catalog_override = {"id": catalog_id, field_name: field_value}
    sparse_catalog_yaml = yaml.dump(
        {"catalogs": [catalog_override]},
        default_flow_style=False,
    )

    # Write sparse override to custom ConfigMap
    sources_cm = ConfigMap(
        name=DEFAULT_CUSTOM_MODEL_CATALOG,
        client=admin_client,
        namespace=model_registry_namespace,
    )
    patches = {"data": {"sources.yaml": sparse_catalog_yaml}}

    with ResourceEditor(patches={sources_cm: patches}):
        is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
        wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)
        yield {
            "catalog_id": catalog_id,
            "field_name": field_name,
            "field_value": field_value,
            "original_catalog": original_catalog,
        }

    is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
    wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)
