from collections.abc import Generator

import pytest
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.resource import ResourceEditor

from tests.model_registry.mcp_servers.config.utils import get_mcp_catalog_sources
from tests.model_registry.mcp_servers.constants import (
    MCP_CATALOG_INVALID_SOURCE,
    MCP_CATALOG_SOURCE,
    MCP_CATALOG_SOURCE2,
    MCP_CATALOG_SOURCE3,
    MCP_CATALOG_SOURCE_ID,
    MCP_CATALOG_SOURCE_NAME,
    MCP_SERVERS_YAML,
    MCP_SERVERS_YAML2,
    MCP_SERVERS_YAML3,
    MCP_SERVERS_YAML_CATALOG_PATH,
)
from tests.model_registry.utils import (
    wait_for_mcp_catalog_api,
    wait_for_model_catalog_pod_ready_after_deletion,
)
from utilities.opendatahub_logger import get_logger

LOGGER = get_logger(name=__name__)


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
    catalog_config_map, current_data = get_mcp_catalog_sources(
        admin_client=admin_client, model_registry_namespace=model_registry_namespace
    )
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
def mcp_source_label_configmap_patch(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    mcp_catalog_rest_urls: list[str],
    model_registry_rest_headers: dict[str, str],
) -> Generator[None]:
    """
    Class-scoped fixture that patches the model-catalog-sources ConfigMap
    with three MCP catalog sources: two labeled and one unlabeled.
    Used for sourceLabel filtering tests (TC-API-036 to TC-API-039).
    """
    catalog_config_map, current_data = get_mcp_catalog_sources(
        admin_client=admin_client, model_registry_namespace=model_registry_namespace
    )
    if "mcp_catalogs" not in current_data:
        current_data["mcp_catalogs"] = []
    current_data["mcp_catalogs"].extend([MCP_CATALOG_SOURCE, MCP_CATALOG_SOURCE2, MCP_CATALOG_SOURCE3])

    patches = {
        "data": {
            "sources.yaml": yaml.dump(current_data, default_flow_style=False),
            "mcp-servers.yaml": MCP_SERVERS_YAML,
            "mcp-servers-2.yaml": MCP_SERVERS_YAML2,
            "mcp-servers-3.yaml": MCP_SERVERS_YAML3,
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
    catalog_config_map, current_data = get_mcp_catalog_sources(
        admin_client=admin_client, model_registry_namespace=model_registry_namespace
    )
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


@pytest.fixture(scope="class")
def mcp_included_excluded_configmap_patch(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    mcp_catalog_rest_urls: list[str],
    model_registry_rest_headers: dict[str, str],
) -> Generator[None]:
    """
    Class-scoped fixture that patches the ConfigMap with an MCP source
    including includedServers/excludedServers filters.

    Parametrized via request.param with a dict containing:
      - "includedServers": list[str] (optional)
      - "excludedServers": list[str] (optional)
    """
    filter_params = request.param

    source_config: dict = {
        "name": MCP_CATALOG_SOURCE_NAME,
        "id": MCP_CATALOG_SOURCE_ID,
        "type": "yaml",
        "enabled": True,
        "properties": {"yamlCatalogPath": MCP_SERVERS_YAML_CATALOG_PATH},
        "labels": [MCP_CATALOG_SOURCE_NAME],
    }
    if "includedServers" in filter_params:
        source_config["includedServers"] = filter_params["includedServers"]
    if "excludedServers" in filter_params:
        source_config["excludedServers"] = filter_params["excludedServers"]

    catalog_config_map, current_data = get_mcp_catalog_sources(
        admin_client=admin_client, model_registry_namespace=model_registry_namespace
    )
    if "mcp_catalogs" not in current_data:
        current_data["mcp_catalogs"] = []
    current_data["mcp_catalogs"].append(source_config)

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
        wait_for_mcp_catalog_api(url=mcp_catalog_rest_urls[0], headers=model_registry_rest_headers)
        yield

    wait_for_model_catalog_pod_ready_after_deletion(
        client=admin_client, model_registry_namespace=model_registry_namespace
    )
