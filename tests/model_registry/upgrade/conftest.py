from typing import Generator

import pytest
from kubernetes.dynamic import DynamicClient

from ocp_resources.config_map import ConfigMap
from ocp_resources.resource import ResourceEditor
from tests.model_registry.utils import (
    is_model_catalog_ready,
    wait_for_model_catalog_api,
    get_custom_model_catalog_cm_data,
)


@pytest.fixture(scope="class")
def pre_upgrade_config_map_update(
    request: pytest.FixtureRequest,
    catalog_config_map: ConfigMap,
    model_registry_namespace: str,
    admin_client: DynamicClient,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> ConfigMap:
    """Fixture for updating catalog config map before upgrade"""
    ResourceEditor(
        patches={
            catalog_config_map: get_custom_model_catalog_cm_data(
                catalog_config_map=catalog_config_map, param=request.param
            )
        }
    ).update()
    is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
    wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)
    return catalog_config_map


@pytest.fixture(scope="class")
def post_upgrade_config_map_update(
    catalog_config_map: ConfigMap,
    admin_client: DynamicClient,
    model_registry_namespace: str,
) -> Generator[ConfigMap, None, None]:
    """Fixture for updating catalog config map after post upgrade testing is done"""
    yield catalog_config_map
    # Only teardown is needed
    catalog_config_map.delete()
    is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
