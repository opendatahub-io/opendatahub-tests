from collections.abc import Generator

import pytest
import requests
import yaml
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.config_map import ConfigMap
from ocp_resources.resource import ResourceEditor
from ocp_resources.route import Route
from simple_logger.logger import get_logger
from timeout_sampler import retry

from tests.model_registry.constants import DEFAULT_CUSTOM_MODEL_CATALOG
from tests.model_registry.mcp_servers.constants import (
    MCP_CATALOG_API_PATH,
    MCP_CATALOG_INVALID_SOURCE,
    MCP_CATALOG_SOURCE,
    MCP_CATALOG_SOURCE2,
    MCP_SERVERS_YAML,
    MCP_SERVERS_YAML2,
)
from tests.model_registry.utils import (
    TransientUnauthorizedError,
    execute_get_call,
    wait_for_model_catalog_pod_ready_after_deletion,
)

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="class")
def mcp_catalog_rest_url(model_registry_namespace: str, model_catalog_routes: list[Route]) -> list[str]:
    """Build MCP catalog REST URL from existing model catalog routes."""
    assert model_catalog_routes, f"Model catalog routes do not exist in {model_registry_namespace}"
    return [f"https://{route.instance.spec.host}:443{MCP_CATALOG_API_PATH}" for route in model_catalog_routes]


@retry(
    wait_timeout=90,
    sleep=5,
    exceptions_dict={ResourceNotFoundError: [], TransientUnauthorizedError: []},
)
def wait_for_mcp_catalog_api(url: str, headers: dict[str, str]) -> requests.Response:
    """Wait for MCP catalog API to be ready and returning MCP server data."""
    LOGGER.info(f"Waiting for MCP catalog API at {url}mcp_servers")
    response = execute_get_call(url=f"{url}mcp_servers", headers=headers)
    data = response.json()
    if not data.get("items"):
        raise ResourceNotFoundError("MCP catalog API returned empty items, catalog data not yet loaded")
    return response


@pytest.fixture(scope="class")
def mcp_servers_configmap_patch(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    mcp_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> Generator[None]:
    """
    Class-scoped fixture that patches the model-catalog-sources ConfigMap

    Sets two keys in the ConfigMap data:
    - sources.yaml: catalog source definition pointing to the MCP servers YAML
    - mcp-servers.yaml: the actual MCP server definitions
    """
    catalog_config_map = ConfigMap(
        name=DEFAULT_CUSTOM_MODEL_CATALOG,
        client=admin_client,
        namespace=model_registry_namespace,
    )

    current_data = yaml.safe_load(catalog_config_map.instance.data.get("sources.yaml", "{}") or "{}")
    if "mcp_catalogs" not in current_data:
        current_data["mcp_catalogs"] = []
    current_data["mcp_catalogs"].append(MCP_CATALOG_SOURCE)

    patches = {
        "data": {
            "sources.yaml": yaml.dump(current_data, default_flow_style=False),
            "mcp-servers.yaml": MCP_SERVERS_YAML,
        }
    }

    with ResourceEditor(patches={catalog_config_map: patches}):
        wait_for_model_catalog_pod_ready_after_deletion(
            client=admin_client, model_registry_namespace=model_registry_namespace
        )
        wait_for_mcp_catalog_api(url=mcp_catalog_rest_url[0], headers=model_registry_rest_headers)
        yield

    wait_for_model_catalog_pod_ready_after_deletion(
        client=admin_client, model_registry_namespace=model_registry_namespace
    )


@pytest.fixture(scope="class")
def mcp_multi_source_configmap_patch(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    mcp_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> Generator[None]:
    """
    Class-scoped fixture that patches the model-catalog-sources ConfigMap
    with two MCP catalog sources pointing to two different YAML files.
    """
    catalog_config_map = ConfigMap(
        name=DEFAULT_CUSTOM_MODEL_CATALOG,
        client=admin_client,
        namespace=model_registry_namespace,
    )

    current_data = yaml.safe_load(catalog_config_map.instance.data.get("sources.yaml", "{}") or "{}")
    if "mcp_catalogs" not in current_data:
        current_data["mcp_catalogs"] = []
    current_data["mcp_catalogs"].extend([MCP_CATALOG_SOURCE, MCP_CATALOG_SOURCE2])

    patches = {
        "data": {
            "sources.yaml": yaml.dump(current_data, default_flow_style=False),
            "mcp-servers.yaml": MCP_SERVERS_YAML,
            "mcp-servers-2.yaml": MCP_SERVERS_YAML2,
        }
    }

    with ResourceEditor(patches={catalog_config_map: patches}):
        wait_for_model_catalog_pod_ready_after_deletion(
            client=admin_client, model_registry_namespace=model_registry_namespace
        )
        wait_for_mcp_catalog_api(url=mcp_catalog_rest_url[0], headers=model_registry_rest_headers)
        yield

    wait_for_model_catalog_pod_ready_after_deletion(
        client=admin_client, model_registry_namespace=model_registry_namespace
    )


@pytest.fixture(scope="class")
def mcp_invalid_yaml_configmap_patch(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    mcp_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> Generator[None]:
    """
    Class-scoped fixture that patches the ConfigMap with a valid MCP source
    plus an invalid one (parameterized via request.param as the invalid YAML content).
    """
    catalog_config_map = ConfigMap(
        name=DEFAULT_CUSTOM_MODEL_CATALOG,
        client=admin_client,
        namespace=model_registry_namespace,
    )

    current_data = yaml.safe_load(catalog_config_map.instance.data.get("sources.yaml", "{}") or "{}")
    if "mcp_catalogs" not in current_data:
        current_data["mcp_catalogs"] = []
    current_data["mcp_catalogs"].extend([MCP_CATALOG_SOURCE, MCP_CATALOG_INVALID_SOURCE])

    patches = {
        "data": {
            "sources.yaml": yaml.dump(current_data, default_flow_style=False),
            "mcp-servers.yaml": MCP_SERVERS_YAML,
            "mcp-servers-invalid.yaml": request.param,
        }
    }

    with ResourceEditor(patches={catalog_config_map: patches}):
        wait_for_model_catalog_pod_ready_after_deletion(
            client=admin_client, model_registry_namespace=model_registry_namespace
        )
        wait_for_mcp_catalog_api(url=mcp_catalog_rest_url[0], headers=model_registry_rest_headers)
        yield

    wait_for_model_catalog_pod_ready_after_deletion(
        client=admin_client, model_registry_namespace=model_registry_namespace
    )
