from collections.abc import Generator
from typing import Any

import pytest
import structlog
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.resource import ResourceEditor
from ocp_resources.route import Route

from tests.ai_hub.agent_catalog.config.constants import (
    TEST_AGENT_CATALOG_LABEL,
    TEST_AGENT_CATALOG_SOURCE,
    TEST_AGENT_CATALOG_SOURCE_ID,
    TEST_AGENT_COUNT,
    TEST_AGENT_LABEL_DEFINITION,
    TEST_AGENTS_YAML,
    TEST_LANGGRAPH_AGENT_NAMES,
)
from tests.ai_hub.agent_catalog.constants import LANGGRAPH_FRAMEWORK
from tests.ai_hub.agent_catalog.utils import get_agent_catalog_sources
from tests.ai_hub.constants import AGENT_CATALOG_API_PATH
from tests.ai_hub.utils import (
    execute_get_command_with_retry,
    wait_for_agent_catalog_api,
    wait_for_model_catalog_pod_ready_after_deletion,
)

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="class")
def agent_catalog_rest_urls(model_registry_namespace: str, model_catalog_routes: list[Route]) -> list[str]:
    """Build agent catalog REST URL from existing model catalog routes."""
    assert model_catalog_routes, f"Model catalog routes do not exist in {model_registry_namespace}"
    return [f"https://{route.instance.spec.host}:443{AGENT_CATALOG_API_PATH}" for route in model_catalog_routes]


@pytest.fixture(scope="class")
def agent_catalog_configmap_patch(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    agent_catalog_rest_urls: list[str],
    model_registry_rest_headers: dict[str, str],
) -> Generator[None]:
    """Patch the catalog sources ConfigMap with a test agent catalog source.

    Accepts config via request.param dict containing:
    - source: catalog source dict (name, id, type, properties, labels)
    - label: source label name
    - label_definition: label definition dict
    - agents_yaml: YAML content for the agents catalog file
    - min_agents: minimum agent count to wait for after patching

    If not parametrized, uses the default custom agent test constants.
    """
    param = getattr(request, "param", None) or {
        "source": TEST_AGENT_CATALOG_SOURCE,
        "label": TEST_AGENT_CATALOG_LABEL,
        "label_definition": TEST_AGENT_LABEL_DEFINITION,
        "agents_yaml": TEST_AGENTS_YAML,
        "min_agents": TEST_AGENT_COUNT,
    }

    source = param["source"]
    catalog_config_map, current_data = get_agent_catalog_sources(
        admin_client=admin_client, model_registry_namespace=model_registry_namespace
    )
    if "agent_catalogs" not in current_data:
        current_data["agent_catalogs"] = []
    current_data["agent_catalogs"] = [
        entry for entry in current_data["agent_catalogs"] if entry.get("id") != source["id"]
    ]
    current_data["agent_catalogs"].append(source)

    labels = current_data.get("labels", [])
    if not any(label.get("name") == param["label"] for label in labels):
        labels.append(param["label_definition"])
    current_data["labels"] = labels

    patches = {
        "data": {
            "sources.yaml": yaml.dump(current_data, default_flow_style=False),
            source["properties"]["yamlCatalogPath"]: param["agents_yaml"],
        }
    }

    with ResourceEditor(patches={catalog_config_map: patches}):
        wait_for_model_catalog_pod_ready_after_deletion(
            client=admin_client, model_registry_namespace=model_registry_namespace
        )
        wait_for_agent_catalog_api(
            url=agent_catalog_rest_urls[0],
            headers=model_registry_rest_headers,
            min_agents=param["min_agents"],
        )
        yield

    wait_for_model_catalog_pod_ready_after_deletion(
        client=admin_client, model_registry_namespace=model_registry_namespace
    )
    wait_for_agent_catalog_api(
        url=agent_catalog_rest_urls[0],
        headers=model_registry_rest_headers,
        min_agents=0,
    )


@pytest.fixture(scope="class")
def agents_response(
    agent_catalog_configmap_patch: None,
    agent_catalog_rest_urls: list[str],
    model_registry_rest_headers: dict[str, str],
) -> dict:
    """Class-scoped fixture that fetches the agents list once per test class."""
    response = execute_get_command_with_retry(
        url=f"{agent_catalog_rest_urls[0]}agents",
        headers=model_registry_rest_headers,
        params={"pageSize": 1000},
    )
    size = response.get("size", 0)
    assert size >= TEST_AGENT_COUNT, f"Expected at least {TEST_AGENT_COUNT} agents after catalog patch, got size={size}"
    return response


@pytest.fixture(scope="class")
def default_agents(
    agent_catalog_rest_urls: list[str],
    model_registry_rest_headers: dict[str, str],
) -> list[dict[str, Any]]:
    """Fetch all agents from the default catalog."""
    response = execute_get_command_with_retry(
        url=f"{agent_catalog_rest_urls[0]}agents",
        headers=model_registry_rest_headers,
        params={"pageSize": 1000},
    )
    items = response.get("items", [])
    assert items, "Expected at least one agent in the default catalog"
    return items


@pytest.fixture(scope="class")
def expected_langgraph_agent_names(agents_response: dict) -> set[str]:
    """Return agent names from the test catalog that use the LangGraph framework."""
    names = {
        agent["name"] for agent in agents_response.get("items", []) if agent.get("framework") == LANGGRAPH_FRAMEWORK
    }
    assert names, f"No agents with framework '{LANGGRAPH_FRAMEWORK}' found in catalog"
    assert TEST_LANGGRAPH_AGENT_NAMES <= names, (
        f"Expected test langgraph agents {TEST_LANGGRAPH_AGENT_NAMES} to be in {names}"
    )
    return names


@pytest.fixture(scope="class")
def model_catalog_api_url(model_catalog_routes: list[Route]) -> str:
    """Base URL for the model catalog API used by the shared /sources endpoint."""
    assert model_catalog_routes, "Model catalog routes not found"
    return f"https://{model_catalog_routes[0].instance.spec.host}:443/api/model_catalog/v1alpha1/"


@pytest.fixture(scope="class")
def configured_agent_catalog_ids(agent_catalog_configmap_patch: None) -> set[str]:
    """Return agent catalog source IDs configured by the test fixture."""
    return {TEST_AGENT_CATALOG_SOURCE_ID}
