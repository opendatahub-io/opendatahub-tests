from collections.abc import Generator

import pytest
import structlog
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.resource import ResourceEditor

from tests.model_registry.mcp_servers.config.constants import (
    EXPECTED_DEFAULT_MCP_CATALOG,
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
def random_default_mcp_server(default_mcp_servers: dict) -> dict:
    """Return the first MCP server from the default catalog response."""
    return default_mcp_servers["items"][0]


@pytest.fixture(scope="class")
def random_default_mcp_server_tools(
    mcp_catalog_rest_urls: list[str],
    model_registry_rest_headers: dict[str, str],
    random_default_mcp_server: dict,
) -> list[dict]:
    """Return the tools for the first default MCP server, asserting toolCount matches."""
    server_id = random_default_mcp_server["id"]
    response = execute_get_command(
        url=f"{mcp_catalog_rest_urls[0]}mcp_servers/{server_id}",
        headers=model_registry_rest_headers,
        params={"includeTools": "true"},
    )
    tool_count = response.get("toolCount", 0)
    tools = response.get("tools", [])
    assert len(tools) == tool_count, (
        f"Tool count mismatch for server '{server_id}': toolCount={tool_count}, actual tools={len(tools)}"
    )
    return tools


@pytest.fixture(scope="class")
def random_default_mcp_server_tool(random_default_mcp_server_tools: list[dict]) -> dict:
    """Return a random tool from the first default MCP server."""
    return random_default_mcp_server_tools[0]


@pytest.fixture(scope="class")
def disable_default_mcp_source(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    mcp_catalog_rest_urls: list[str],
    model_registry_rest_headers: dict[str, str],
) -> Generator[None]:
    """Class-scoped fixture that disables the default MCP catalog source and restores it after."""
    catalog_config_map, current_data = get_mcp_catalog_sources(
        admin_client=admin_client, model_registry_namespace=model_registry_namespace
    )
    current_data["mcp_catalogs"] = [
        {
            "name": EXPECTED_DEFAULT_MCP_CATALOG["name"],
            "id": EXPECTED_DEFAULT_MCP_CATALOG["id"],
            "enabled": False,
        }
    ]

    patches = {"data": {"sources.yaml": yaml.dump(current_data, default_flow_style=False)}}

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
