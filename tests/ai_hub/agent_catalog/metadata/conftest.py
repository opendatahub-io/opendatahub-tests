from typing import Any

import pytest
import yaml
from kubernetes.dynamic import DynamicClient

from tests.ai_hub.agent_catalog.config.constants import EXPECTED_DEFAULT_AGENT_CATALOG
from tests.ai_hub.constants import CATALOG_CONTAINER
from tests.ai_hub.utils import get_model_catalog_pod


@pytest.fixture(scope="class")
def default_agents_yaml_content(
    admin_client: DynamicClient,
    model_registry_namespace: str,
) -> dict[str, Any]:
    """Fetch and parse the agents catalog YAML from the catalog pod."""
    catalog_path = EXPECTED_DEFAULT_AGENT_CATALOG["properties"]["yamlCatalogPath"]
    pods = get_model_catalog_pod(client=admin_client, model_registry_namespace=model_registry_namespace)
    assert pods, "No catalog pods found"
    raw = pods[0].execute(command=["cat", catalog_path], container=CATALOG_CONTAINER)
    return yaml.safe_load(raw)
