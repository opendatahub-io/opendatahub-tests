from collections.abc import Generator

import pytest
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.resource import ResourceEditor
from simple_logger.logger import get_logger

from tests.model_registry.constants import DEFAULT_CUSTOM_MODEL_CATALOG
from tests.model_registry.mcp_servers.constants import (
    MCP_CATALOG_INVALID_SOURCE,
    MCP_CATALOG_SOURCE,
    MCP_CATALOG_SOURCE2,
    MCP_SERVERS_YAML,
    MCP_SERVERS_YAML2,
)
from tests.model_registry.utils import (
    execute_get_command,
    wait_for_mcp_catalog_api,
    wait_for_model_catalog_pod_ready_after_deletion,
)

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="class")
def mcp_servers_response(
    mcp_catalog_rest_urls: list[str],
    model_registry_rest_headers: dict[str, str],
) -> dict:
    """Class-scoped fixture that fetches the MCP servers list once per test class."""
    return execute_get_command(
        url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
        headers=model_registry_rest_headers,
    )


@pytest.fixture(scope="class")
def mcp_multi_source_configmap_patch(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    mcp_catalog_rest_urls: list[str],
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
        wait_for_mcp_catalog_api(url=mcp_catalog_rest_urls[0], headers=model_registry_rest_headers)
        yield

    wait_for_model_catalog_pod_ready_after_deletion(
        client=admin_client, model_registry_namespace=model_registry_namespace
    )


@pytest.fixture(scope="class")
def mcp_invalid_yaml_configmap_patch(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    mcp_catalog_rest_urls: list[str],
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
        wait_for_mcp_catalog_api(url=mcp_catalog_rest_urls[0], headers=model_registry_rest_headers)
        yield

    wait_for_model_catalog_pod_ready_after_deletion(
        client=admin_client, model_registry_namespace=model_registry_namespace
    )
