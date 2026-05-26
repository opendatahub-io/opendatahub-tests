import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.route import Route

from tests.model_registry.constants import MCP_CATALOG_API_PATH
from tests.model_registry.utils import execute_get_command, get_rest_headers

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="session")
def mcp_catalog_rest_urls_scope_session(
    model_registry_namespace: str,
    admin_client: DynamicClient,
) -> list[str]:
    """Session-scoped MCP catalog REST URLs."""
    routes = list(
        Route.get(namespace=model_registry_namespace, label_selector="component=model-catalog", client=admin_client)
    )
    assert routes, f"Model catalog routes do not exist in {model_registry_namespace}"
    return [f"https://{route.instance.spec.host}:443{MCP_CATALOG_API_PATH}" for route in routes]


@pytest.fixture(scope="session")
def model_registry_rest_headers_scope_session(current_client_token: str) -> dict[str, str]:
    """Session-scoped model registry REST headers."""
    return get_rest_headers(token=current_client_token)


@pytest.fixture(scope="session", autouse=True)
def default_mcp_servers(
    mcp_catalog_rest_urls_scope_session: list[str],
    model_registry_rest_headers_scope_session: dict[str, str],
) -> dict:
    """Session-scoped fixture that fetches the default MCP servers list once per session."""
    return execute_get_command(
        url=f"{mcp_catalog_rest_urls_scope_session[0]}mcp_servers",
        headers=model_registry_rest_headers_scope_session,
        params={"pageSize": 1000},
    )


@pytest.fixture(scope="class")
def custom_mcp_servers(mcp_servers_response: dict, default_mcp_servers: dict) -> list[dict]:
    """Return only the custom MCP servers by excluding default servers from the full response."""
    default_server_ids = {server["name"] for server in default_mcp_servers.get("items", [])}
    return [server for server in mcp_servers_response.get("items", []) if server["name"] not in default_server_ids]


@pytest.fixture(scope="class")
def mcp_servers_response(
    mcp_catalog_rest_urls: list[str],
    model_registry_rest_headers: dict[str, str],
) -> dict:
    """Class-scoped fixture that fetches the MCP servers list once per test class."""
    return execute_get_command(
        url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
        headers=model_registry_rest_headers,
        params={"pageSize": 1000},
    )
