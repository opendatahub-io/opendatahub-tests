from typing import Any

import pytest
import structlog
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.route import Route

from tests.ai_hub.constants import CATALOG_CONTAINER
from tests.ai_hub.mcp_servers.preview.utils import build_mcp_preview_config, post_stateless_preview
from tests.ai_hub.utils import execute_authenticated_post, get_model_catalog_pod
from utilities.infra import get_openshift_token

LOGGER = structlog.get_logger(name=__name__)

MCP_CATALOG_FILE: str = "/shared-data/redhat-mcp-servers-catalog.yaml"

MODEL_CATALOG_API_PATH: str = "/api/model_catalog/v1alpha1/"


@pytest.fixture(scope="class")
def model_catalog_preview_url(model_registry_namespace: str, admin_client: DynamicClient) -> str:
    """Base URL for the model catalog API (preview endpoint lives here, not on the MCP catalog API)."""
    routes = list(
        Route.get(namespace=model_registry_namespace, label_selector="component=model-catalog", client=admin_client)
    )
    assert routes, f"Model catalog routes do not exist in {model_registry_namespace}"
    return f"https://{routes[0].instance.spec.host}:443{MODEL_CATALOG_API_PATH}"


@pytest.fixture(scope="class")
def preview_user_token() -> str:
    """Authentication token for preview API calls."""
    return get_openshift_token()


@pytest.fixture(scope="class")
def default_mcp_catalog_yaml_content(
    admin_client: DynamicClient,
    model_registry_namespace: str,
) -> dict[str, Any]:
    """Fetch and parse MCP server catalog YAML from the catalog pod."""
    pods = get_model_catalog_pod(client=admin_client, model_registry_namespace=model_registry_namespace)
    assert pods, f"No model-catalog pods found in {model_registry_namespace}"
    model_catalog_pod = pods[0]
    raw = model_catalog_pod.execute(command=["cat", MCP_CATALOG_FILE], container=CATALOG_CONTAINER)
    return yaml.safe_load(raw)


@pytest.fixture(scope="class")
def default_mcp_servers_from_yaml(default_mcp_catalog_yaml_content: dict[str, Any]) -> list[dict[str, Any]]:
    """MCP server list extracted from the default catalog YAML."""
    servers = default_mcp_catalog_yaml_content.get("mcp_servers", [])
    assert servers, "No mcp_servers found in default MCP catalog YAML"
    return servers


@pytest.fixture()
def mcp_preview_result(
    request: pytest.FixtureRequest,
    model_catalog_preview_url: str,
    preview_user_token: str,
) -> tuple[dict[str, Any], list[str] | None, list[str] | None]:
    """Execute the MCP preview API and return (result, included_patterns, excluded_patterns).

    Accepts parametrize with 'included_servers', 'excluded_servers', 'filter_status' keys.
    """
    param = getattr(request, "param", {})
    included_servers = param.get("included_servers")
    excluded_servers = param.get("excluded_servers")

    config_content = build_mcp_preview_config(
        yaml_catalog_path=MCP_CATALOG_FILE,
        included_servers=included_servers,
        excluded_servers=excluded_servers,
    )

    filter_status = param.get("filter_status", "")
    query = f"pageSize=100&filterStatus={filter_status}" if filter_status else "pageSize=100"
    url = f"{model_catalog_preview_url}sources/preview?{query}"

    files = {"config": ("config.yaml", config_content, "application/x-yaml")}
    result = execute_authenticated_post(url=url, token=preview_user_token, files=files)
    return result, included_servers, excluded_servers


@pytest.fixture()
def stateless_preview_result(
    request: pytest.FixtureRequest,
    model_catalog_preview_url: str,
    preview_user_token: str,
) -> tuple[dict[str, Any], list[str] | None, list[str] | None]:
    """Execute the MCP preview API in stateless mode and return (result, included, excluded)."""
    param = getattr(request, "param", {})
    included_servers = param.get("included_servers")
    excluded_servers = param.get("excluded_servers")
    LOGGER.info(f"Stateless preview: includedServers={included_servers}, excludedServers={excluded_servers}")

    result = post_stateless_preview(
        model_catalog_preview_url=model_catalog_preview_url,
        token=preview_user_token,
        included_servers=included_servers,
        excluded_servers=excluded_servers,
    )
    return result, included_servers, excluded_servers


@pytest.fixture()
def paginated_preview_pages(
    request: pytest.FixtureRequest,
    model_catalog_preview_url: str,
    preview_user_token: str,
) -> list[dict[str, Any]]:
    """Walk all pages of a paginated MCP preview and return the list of page results.

    Accepts parametrize with 'page_size', 'included_servers', 'excluded_servers' keys.
    """
    param = getattr(request, "param", {})
    page_size = param.get("page_size", 2)
    included_servers = param.get("included_servers", ["*"])
    excluded_servers = param.get("excluded_servers")

    LOGGER.info(
        f"Paginating preview: pageSize={page_size}, "
        f"includedServers={included_servers}, excludedServers={excluded_servers}"
    )

    max_pages = 50
    pages: list[dict[str, Any]] = []
    next_token = ""

    while len(pages) < max_pages:
        result = post_stateless_preview(
            model_catalog_preview_url=model_catalog_preview_url,
            token=preview_user_token,
            page_size=page_size,
            next_page_token=next_token,
            included_servers=included_servers,
            excluded_servers=excluded_servers,
        )
        pages.append(result)

        page_names = [item["name"] for item in result.get("items", [])]
        LOGGER.info(
            f"Page {len(pages)}: servers={page_names}, "
            f"nextPageToken={result.get('nextPageToken', '')!r}, summary={result.get('summary')}"
        )

        next_token = result.get("nextPageToken", "")
        if not next_token:
            break

    LOGGER.info(f"Pagination complete: {len(pages)} pages collected")
    return pages
