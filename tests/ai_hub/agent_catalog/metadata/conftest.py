import json
from typing import Any

import pytest
import yaml
from kubernetes.dynamic import DynamicClient

from tests.ai_hub.agent_catalog.config.constants import EXPECTED_DEFAULT_AGENT_CATALOG
from tests.ai_hub.constants import CATALOG_CONTAINER
from tests.ai_hub.utils import execute_get_command_with_retry, get_model_catalog_pod


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


@pytest.fixture(scope="class")
def agent_template_artifacts(
    agent_catalog_rest_urls: list[str],
    model_registry_rest_headers: dict[str, str],
    default_agents: list[dict[str, Any]],
) -> dict[str, dict]:
    """Fetch template artifacts for all default agents.

    Returns:
        dict mapping agent name to its parsed agent.yaml template content (dict).
        Agents without a template artifact are omitted.
    """
    templates: dict[str, dict] = {}
    for agent in default_agents:
        response = execute_get_command_with_retry(
            url=f"{agent_catalog_rest_urls[0]}agents/{agent['id']}/artifacts",
            headers=model_registry_rest_headers,
            params={"artifactType": "template-artifact", "pageSize": 1000},
        )
        template = next(
            (item for item in response.get("items", []) if item.get("name") == "agent.yaml"),
            None,
        )
        if template and template.get("content"):
            templates[agent["name"]] = json.loads(template["content"])
    return templates
