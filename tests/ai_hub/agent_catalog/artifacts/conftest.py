from collections.abc import Generator

import pytest
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.resource import ResourceEditor

from tests.ai_hub.agent_catalog.artifacts.constants import (
    ARTIFACT_DEFAULT_NAME_AGENT_NAME,
    ARTIFACT_EMPTY_AGENT_NAME,
    ARTIFACT_FULL_AGENT_NAME,
    ARTIFACT_IMAGE_ONLY_AGENT_NAME,
    ARTIFACT_TEST_AGENTS_YAML,
    ARTIFACT_TEST_LABEL,
    ARTIFACT_TEST_LABEL_DEFINITION,
    ARTIFACT_TEST_SOURCE,
    ARTIFACT_TEST_SOURCE_ID,
    ARTIFACT_TEST_YAML_PATH,
)
from tests.ai_hub.agent_catalog.utils import get_agent_catalog_sources
from tests.ai_hub.utils import (
    execute_get_command_with_retry,
    wait_for_agent_catalog_api,
    wait_for_model_catalog_pod_ready_after_deletion,
)


@pytest.fixture(scope="class")
def artifact_agent_configmap_patch(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    agent_catalog_rest_urls: list[str],
    model_registry_rest_headers: dict[str, str],
) -> Generator[None]:
    """Patch the catalog sources ConfigMap with a test agent that has artifacts and templates."""
    catalog_config_map, current_data = get_agent_catalog_sources(
        admin_client=admin_client, model_registry_namespace=model_registry_namespace
    )
    if "agent_catalogs" not in current_data:
        current_data["agent_catalogs"] = []
    current_data["agent_catalogs"] = [
        entry for entry in current_data["agent_catalogs"] if entry.get("id") != ARTIFACT_TEST_SOURCE_ID
    ]
    current_data["agent_catalogs"].append(ARTIFACT_TEST_SOURCE)

    labels = current_data.get("labels", [])
    if not any(label.get("name") == ARTIFACT_TEST_LABEL for label in labels):
        labels.append(ARTIFACT_TEST_LABEL_DEFINITION)
    current_data["labels"] = labels

    patches = {
        "data": {
            "sources.yaml": yaml.dump(current_data, default_flow_style=False),
            ARTIFACT_TEST_YAML_PATH: ARTIFACT_TEST_AGENTS_YAML,
        }
    }

    with ResourceEditor(patches={catalog_config_map: patches}):
        wait_for_model_catalog_pod_ready_after_deletion(
            client=admin_client, model_registry_namespace=model_registry_namespace
        )
        wait_for_agent_catalog_api(
            url=agent_catalog_rest_urls[0],
            headers=model_registry_rest_headers,
            min_agents=4,
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


def _get_agent_id_by_name(
    agent_catalog_rest_urls: list[str],
    model_registry_rest_headers: dict[str, str],
    agent_name: str,
) -> str:
    """Look up an agent ID by name from the catalog API."""
    response = execute_get_command_with_retry(
        url=f"{agent_catalog_rest_urls[0]}agents",
        headers=model_registry_rest_headers,
        params={"pageSize": 1000},
    )
    agent = next(
        (item for item in response.get("items", []) if item["name"] == agent_name),
        None,
    )
    assert agent, f"Agent '{agent_name}' not found after ConfigMap patch"
    return agent["id"]


@pytest.fixture(scope="class")
def artifact_agent_id(
    artifact_agent_configmap_patch: None,
    agent_catalog_rest_urls: list[str],
    model_registry_rest_headers: dict[str, str],
) -> str:
    """Return the ID of the test agent that has both image and template artifacts."""
    return _get_agent_id_by_name(
        agent_catalog_rest_urls=agent_catalog_rest_urls,
        model_registry_rest_headers=model_registry_rest_headers,
        agent_name=ARTIFACT_FULL_AGENT_NAME,
    )


@pytest.fixture(scope="class")
def empty_artifact_agent_id(
    artifact_agent_configmap_patch: None,
    agent_catalog_rest_urls: list[str],
    model_registry_rest_headers: dict[str, str],
) -> str:
    """Return the ID of the test agent with no artifacts."""
    return _get_agent_id_by_name(
        agent_catalog_rest_urls=agent_catalog_rest_urls,
        model_registry_rest_headers=model_registry_rest_headers,
        agent_name=ARTIFACT_EMPTY_AGENT_NAME,
    )


@pytest.fixture(scope="class")
def image_only_artifact_agent_id(
    artifact_agent_configmap_patch: None,
    agent_catalog_rest_urls: list[str],
    model_registry_rest_headers: dict[str, str],
) -> str:
    """Return the ID of the test agent with only image artifacts."""
    return _get_agent_id_by_name(
        agent_catalog_rest_urls=agent_catalog_rest_urls,
        model_registry_rest_headers=model_registry_rest_headers,
        agent_name=ARTIFACT_IMAGE_ONLY_AGENT_NAME,
    )


@pytest.fixture(scope="class")
def default_name_artifact_agent_id(
    artifact_agent_configmap_patch: None,
    agent_catalog_rest_urls: list[str],
    model_registry_rest_headers: dict[str, str],
) -> str:
    """Return the ID of the test agent with an unnamed template."""
    return _get_agent_id_by_name(
        agent_catalog_rest_urls=agent_catalog_rest_urls,
        model_registry_rest_headers=model_registry_rest_headers,
        agent_name=ARTIFACT_DEFAULT_NAME_AGENT_NAME,
    )
