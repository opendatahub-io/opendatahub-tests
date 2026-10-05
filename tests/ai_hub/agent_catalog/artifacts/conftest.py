import pytest

from tests.ai_hub.agent_catalog.artifacts.constants import (
    ARTIFACT_DEFAULT_NAME_AGENT_NAME,
    ARTIFACT_EMPTY_AGENT_NAME,
    ARTIFACT_FULL_AGENT_NAME,
    ARTIFACT_IMAGE_ONLY_AGENT_NAME,
)
from tests.ai_hub.agent_catalog.artifacts.utils import get_agent_id_by_name


@pytest.fixture(scope="class")
def artifact_agent_id(
    agent_catalog_configmap_patch: None,
    agent_catalog_rest_urls: list[str],
    model_registry_rest_headers: dict[str, str],
) -> str:
    """Return the ID of the test agent that has both image and template artifacts."""
    return get_agent_id_by_name(
        agent_catalog_rest_urls=agent_catalog_rest_urls,
        model_registry_rest_headers=model_registry_rest_headers,
        agent_name=ARTIFACT_FULL_AGENT_NAME,
    )


@pytest.fixture(scope="class")
def empty_artifact_agent_id(
    agent_catalog_configmap_patch: None,
    agent_catalog_rest_urls: list[str],
    model_registry_rest_headers: dict[str, str],
) -> str:
    """Return the ID of the test agent with no artifacts."""
    return get_agent_id_by_name(
        agent_catalog_rest_urls=agent_catalog_rest_urls,
        model_registry_rest_headers=model_registry_rest_headers,
        agent_name=ARTIFACT_EMPTY_AGENT_NAME,
    )


@pytest.fixture(scope="class")
def image_only_artifact_agent_id(
    agent_catalog_configmap_patch: None,
    agent_catalog_rest_urls: list[str],
    model_registry_rest_headers: dict[str, str],
) -> str:
    """Return the ID of the test agent with only image artifacts."""
    return get_agent_id_by_name(
        agent_catalog_rest_urls=agent_catalog_rest_urls,
        model_registry_rest_headers=model_registry_rest_headers,
        agent_name=ARTIFACT_IMAGE_ONLY_AGENT_NAME,
    )


@pytest.fixture(scope="class")
def default_name_artifact_agent_id(
    agent_catalog_configmap_patch: None,
    agent_catalog_rest_urls: list[str],
    model_registry_rest_headers: dict[str, str],
) -> str:
    """Return the ID of the test agent with an unnamed template."""
    return get_agent_id_by_name(
        agent_catalog_rest_urls=agent_catalog_rest_urls,
        model_registry_rest_headers=model_registry_rest_headers,
        agent_name=ARTIFACT_DEFAULT_NAME_AGENT_NAME,
    )
