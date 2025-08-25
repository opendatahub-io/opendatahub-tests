from typing import Generator

import pytest
from kubernetes.dynamic import DynamicClient

from ocp_resources.config_map import ConfigMap
from ocp_resources.resource import ResourceEditor

from ocp_resources.route import Route
from tests.model_registry.model_catalog.constants import DEFAULT_MODEL_CATALOG
from tests.model_registry.model_catalog.utils import is_model_catalog_ready, wait_for_model_catalog_api


@pytest.fixture(scope="class")
def catalog_config_map(admin_client: DynamicClient, model_registry_namespace: str) -> ConfigMap:
    return ConfigMap(name=DEFAULT_MODEL_CATALOG, client=admin_client, namespace=model_registry_namespace)


@pytest.fixture(scope="class")
def model_catalog_routes(admin_client: DynamicClient, model_registry_namespace: str) -> list[Route]:
    return list(
        Route.get(namespace=model_registry_namespace, label_selector="component=model-catalog", dyn_client=admin_client)
    )


@pytest.fixture(scope="class")
def model_catalog_rest_url(model_registry_namespace: str, model_catalog_routes: list[Route]) -> list[str]:
    assert model_catalog_routes, f"Model catalog routes does not exist in {model_registry_namespace}"
    route_urls = [
        f"https://{route.instance.spec.host}:443/api/model_catalog/v1alpha1/" for route in model_catalog_routes
    ]
    assert route_urls, (
        "Model catalog routes information could not be found from "
        f"routes:{[route.name for route in model_catalog_routes]}"
    )
    return route_urls


@pytest.fixture(scope="class")
def updated_catalog_config_map(
    request: pytest.FixtureRequest,
    catalog_config_map: ConfigMap,
    model_registry_namespace: str,
    admin_client: DynamicClient,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> Generator[ConfigMap, None, None]:
    patches = {"data": {"sources.yaml": request.param["sources_yaml"]}}
    if "sample_yaml" in request.param:
        patches["data"]["sample-catalog.yaml"] = request.param["sample_yaml"]

    with ResourceEditor(patches={catalog_config_map: patches}):
        is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
        wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)
        yield catalog_config_map
    is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)


@pytest.fixture(scope="class")
def expected_catalog_values(request: pytest.FixtureRequest) -> dict[str, str]:
    return request.param
