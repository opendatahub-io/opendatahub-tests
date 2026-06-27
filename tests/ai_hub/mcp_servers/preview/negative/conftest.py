import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.route import Route

from utilities.infra import get_openshift_token

LOGGER = structlog.get_logger(name=__name__)

CATALOG_CONTAINER: str = "catalog"
MCP_CATALOG_FILE: str = "/shared-data/redhat-mcp-servers-catalog.yaml"

MODEL_CATALOG_API_PATH: str = "/api/model_catalog/v1alpha1/"


@pytest.fixture(scope="class")
def preview_user_token() -> str:
    """Authentication token for preview API calls."""
    return get_openshift_token()


@pytest.fixture(scope="class")
def model_catalog_preview_url(model_registry_namespace: str, admin_client: DynamicClient) -> str:
    """Model catalog REST URL for the sources/preview endpoint."""
    routes = list(
        Route.get(namespace=model_registry_namespace, label_selector="component=model-catalog", client=admin_client)
    )
    assert routes, f"Model catalog routes do not exist in {model_registry_namespace}"
    return f"https://{routes[0].instance.spec.host}:443{MODEL_CATALOG_API_PATH}"
