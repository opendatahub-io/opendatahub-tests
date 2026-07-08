from typing import Any

import pytest
import yaml
from kubernetes.dynamic import DynamicClient

from tests.ai_hub.constants import CATALOG_CONTAINER
from tests.ai_hub.utils import get_model_catalog_pod

AGENTS_CATALOG_FILE: str = "/shared-data/redhat-agents-catalog.yaml"


@pytest.fixture(scope="class")
def default_agents_yaml_content(
    admin_client: DynamicClient,
    model_registry_namespace: str,
) -> dict[str, Any]:
    """Fetch and parse the agents catalog YAML from the catalog pod."""
    pods = get_model_catalog_pod(client=admin_client, model_registry_namespace=model_registry_namespace)
    assert pods, "No catalog pods found"
    raw = pods[0].execute(command=["cat", AGENTS_CATALOG_FILE], container=CATALOG_CONTAINER)
    return yaml.safe_load(raw)
