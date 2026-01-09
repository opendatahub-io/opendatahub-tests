import pytest
import yaml
from ocp_resources.config_map import ConfigMap
from ocp_resources.resource import ResourceEditor
from simple_logger.logger import get_logger
from typing import Generator

from tests.model_registry.constants import DEFAULT_CUSTOM_MODEL_CATALOG
from tests.model_registry.model_catalog.constants import REDHAT_AI_CATALOG_ID
from tests.model_registry.utils import (
    execute_get_command,
    is_model_catalog_ready,
    wait_for_model_catalog_api,
)

LOGGER = get_logger(name=__name__)


@pytest.fixture()
def sparse_override_catalog_source(
    admin_client,
    model_registry_namespace: str,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> Generator[dict, None, None]:
    """
    Creates a sparse override for an existing default catalog source.
    """
    catalog_id = REDHAT_AI_CATALOG_ID
    custom_name = "Custom Override Name"
    custom_labels = ["custom-label", "override-label"]

    # Capture CURRENT catalog state from API before applying sparse override
    response = execute_get_command(url=f"{model_catalog_rest_url[0]}sources", headers=model_registry_rest_headers)
    items = response.get("items", [])
    original_catalog = next((item for item in items if item.get("id") == catalog_id), None)
    assert original_catalog is not None, f"Original catalog '{catalog_id}' not found in sources"

    LOGGER.info(f"Original catalog state before sparse override: {original_catalog}")

    # Create sparse override YAML with ONLY id, name, and labels
    # Deliberately NOT including some fields that should be inherited from the default ConfigMap
    sparse_catalog_yaml = yaml.dump(
        {
            "catalogs": [
                {
                    "id": catalog_id,
                    "name": custom_name,
                    "labels": custom_labels,
                }
            ]
        },
        default_flow_style=False,
    )

    LOGGER.info(f"Sparse override YAML:\n{sparse_catalog_yaml}")

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
            "custom_name": custom_name,
            "custom_labels": custom_labels,
            "original_catalog": original_catalog,
        }

    is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
    wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)
