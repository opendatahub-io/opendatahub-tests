from collections.abc import Generator

import pytest
import structlog
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.resource import ResourceEditor

from tests.model_registry.constants import DEFAULT_MODEL_CATALOG_CM
from tests.model_registry.mcp_servers.config.constants import (
    MCP_CATALOG_INVALID_SOURCE,
    MCP_CATALOG_SOURCE,
    MCP_CATALOG_SOURCE2,
    MCP_CATALOG_SOURCE_ID,
    MCP_CATALOG_SOURCE_NAME,
    MCP_SERVERS_YAML,
    MCP_SERVERS_YAML2,
    MCP_SERVERS_YAML_CATALOG_PATH,
    NAMED_QUERIES,
)
from tests.model_registry.mcp_servers.config.utils import get_mcp_catalog_sources
from tests.model_registry.utils import (
    execute_get_command,
    wait_for_mcp_catalog_api,
    wait_for_model_catalog_pod_ready_after_deletion,
)

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="class")
def default_catalog_sources_data(
    admin_client: DynamicClient,
    model_registry_namespace: str,
) -> dict:
    """Return the parsed sources.yaml data from the default catalog sources ConfigMap."""

    configmap = ConfigMap(
        name=DEFAULT_MODEL_CATALOG_CM,
        client=admin_client,
        namespace=model_registry_namespace,
    )
    return yaml.safe_load(configmap.instance.data.get("sources.yaml", "{}") or "{}")


@pytest.fixture(scope="class")
def default_mcp_catalogs(default_catalog_sources_data: dict) -> list[dict]:
    """Return the mcp_catalogs list from the default catalog sources ConfigMap."""
    return default_catalog_sources_data.get("mcp_catalogs", [])


@pytest.fixture(scope="class")
def default_mcp_label_definitions(default_catalog_sources_data: dict) -> list[dict]:
    """Return the MCP server label definitions from the default catalog sources ConfigMap."""
    return [
        label for label in default_catalog_sources_data.get("labels", []) if label.get("assetType") == "mcp_servers"
    ]


@pytest.fixture(scope="class")
def mcp_servers_by_source(
    request: pytest.FixtureRequest,
    mcp_catalog_rest_urls: list[str],
    model_registry_rest_headers: dict[str, str],
) -> dict:
    """Return MCP servers filtered by source label, passed via request.param."""
    source_label = request.param
    return execute_get_command(
        url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
        headers=model_registry_rest_headers,
        params={"sourceLabel": source_label, "pageSize": 1000},
    )


@pytest.fixture(scope="class")
def mcp_server_with_multiple_tools(default_mcp_servers: dict) -> tuple[str, int]:
    """Return the ID and tool count of a default MCP server that has at least 2 tools."""
    server = next(
        (server for server in default_mcp_servers.get("items", []) if server.get("toolCount", 0) >= 2),
        None,
    )
    assert server, "No default MCP server found with at least 2 tools"
    return server["id"], server["toolCount"]


@pytest.fixture(scope="class")
def disable_default_mcp_source(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    mcp_catalog_rest_urls: list[str],
    model_registry_rest_headers: dict[str, str],
) -> Generator[str]:
    """Class-scoped fixture that disables a default MCP catalog source and restores it after.

    The catalog to disable is passed via request.param (expected catalog dict).
    Yields the catalog ID of the disabled source.
    """
    expected_catalog = request.param
    catalog_config_map, current_data = get_mcp_catalog_sources(
        admin_client=admin_client, model_registry_namespace=model_registry_namespace
    )
    current_data["mcp_catalogs"] = [
        {
            "name": expected_catalog["name"],
            "id": expected_catalog["id"],
            "enabled": False,
        }
    ]

    patches = {"data": {"sources.yaml": yaml.dump(current_data, default_flow_style=False)}}

    with ResourceEditor(patches={catalog_config_map: patches}):
        wait_for_model_catalog_pod_ready_after_deletion(
            client=admin_client, model_registry_namespace=model_registry_namespace
        )
        wait_for_mcp_catalog_api(url=mcp_catalog_rest_urls[0], headers=model_registry_rest_headers)
        yield expected_catalog["id"]

    wait_for_model_catalog_pod_ready_after_deletion(
        client=admin_client, model_registry_namespace=model_registry_namespace
    )
    wait_for_mcp_catalog_api(url=mcp_catalog_rest_urls[0], headers=model_registry_rest_headers)


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
    wait_for_mcp_catalog_api(url=mcp_catalog_rest_urls[0], headers=model_registry_rest_headers)


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
    wait_for_mcp_catalog_api(url=mcp_catalog_rest_urls[0], headers=model_registry_rest_headers)


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
    wait_for_mcp_catalog_api(url=mcp_catalog_rest_urls[0], headers=model_registry_rest_headers)


@pytest.fixture(scope="class")
def mcp_servers_configmap_patch(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    mcp_catalog_rest_urls: list[str],
    model_registry_rest_headers: dict[str, str],
) -> Generator[None]:
    """
    Class-scoped fixture that patches the model-catalog-sources ConfigMap.

    Sets two keys in the ConfigMap data:
    - sources.yaml: catalog source definition pointing to the MCP servers YAML,
      plus named queries for filtering by custom properties
    - mcp-servers.yaml: the actual MCP server definitions
    """
    catalog_config_map, current_data = get_mcp_catalog_sources(
        admin_client=admin_client, model_registry_namespace=model_registry_namespace
    )
    if "mcp_catalogs" not in current_data:
        current_data["mcp_catalogs"] = []
    current_data["mcp_catalogs"].append(MCP_CATALOG_SOURCE)
    current_data["namedQueries"] = NAMED_QUERIES

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
    wait_for_mcp_catalog_api(url=mcp_catalog_rest_urls[0], headers=model_registry_rest_headers)
