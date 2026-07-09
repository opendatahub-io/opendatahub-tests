import pytest
from ocp_resources.route import Route

from tests.ai_hub.agent_catalog.config.constants import AGENT_CATALOG_API_PATH


@pytest.fixture(scope="class")
def agent_catalog_rest_urls(model_registry_namespace: str, model_catalog_routes: list[Route]) -> list[str]:
    """Build agent catalog REST URL from existing model catalog routes."""
    assert model_catalog_routes, f"Model catalog routes do not exist in {model_registry_namespace}"
    return [f"https://{route.instance.spec.host}:443{AGENT_CATALOG_API_PATH}" for route in model_catalog_routes]
