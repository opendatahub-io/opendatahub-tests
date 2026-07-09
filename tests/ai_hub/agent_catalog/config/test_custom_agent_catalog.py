from typing import Self

import pytest
import structlog

from tests.ai_hub.agent_catalog.config.constants import (
    REQUIRED_AGENT_FIELDS,
    TEST_AGENT_NAMES,
)
from tests.ai_hub.utils import execute_get_command_with_retry

LOGGER = structlog.get_logger(name=__name__)

pytestmark = [
    pytest.mark.tier1,
    pytest.mark.usefixtures(
        "updated_dsc_component_state_scope_session",
        "model_registry_namespace",
        "agent_catalog_configmap_patch",
    ),
]


class TestAgentCatalogApi:
    """Tests for agent catalog API endpoints (RHOAIENG-70683)."""

    def test_agents_list_returns_non_empty_catalog(
        self: Self,
        agents_response: dict,
        expected_langgraph_agent_names: set[str],
    ) -> None:
        """Given the agent catalog is configured
        When listing agents via GET /agents
        Then a non-empty list of agents is returned
        """
        items = agents_response.get("items", [])
        returned_names = {agent["name"] for agent in items}
        LOGGER.info(f"Found {len(items)} agents: {returned_names}")
        assert TEST_AGENT_NAMES <= returned_names, (
            f"Expected test agents {TEST_AGENT_NAMES} in response. Missing: {TEST_AGENT_NAMES - returned_names}"
        )
        assert expected_langgraph_agent_names <= returned_names, (
            f"Expected langgraph agents {expected_langgraph_agent_names} in response. "
            f"Missing: {expected_langgraph_agent_names - returned_names}"
        )

    def test_agents_list_required_fields(
        self: Self,
        agents_response: dict,
    ) -> None:
        """Given agents are listed from the catalog
        When inspecting each agent entry
        Then required metadata fields are present
        """
        errors = []
        for agent in agents_response.get("items", []):
            agent_name = agent["name"]
            for field in REQUIRED_AGENT_FIELDS:
                if not agent.get(field):
                    errors.append(f"Agent '{agent_name}' is missing required field '{field}'")
        assert not errors, "Required field validation failed:\n" + "\n".join(errors)

    def test_get_agent_by_id(
        self: Self,
        agent_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        agents_response: dict,
    ) -> None:
        """Given an agent exists in the list response
        When fetching that agent by ID
        Then the response matches the list entry
        """
        agent = agents_response["items"][0]
        agent_id = agent["id"]
        fetched = execute_get_command_with_retry(
            url=f"{agent_catalog_rest_urls[0]}agents/{agent_id}",
            headers=model_registry_rest_headers,
        )
        LOGGER.info(f"Fetched agent by id: {agent_id}")
        assert fetched == agent, (
            f"Agent fetched by id does not match list response.\nExpected: {agent}\nActual: {fetched}"
        )

    def test_sources_asset_type_agents(
        self: Self,
        model_catalog_api_url: str,
        model_registry_rest_headers: dict[str, str],
        configured_agent_catalog_ids: set[str],
    ) -> None:
        """Given agent catalog sources are configured
        When requesting GET /sources?assetType=agents
        Then configured agent sources are returned with at least one available
        """
        response = execute_get_command_with_retry(
            url=f"{model_catalog_api_url}sources",
            headers=model_registry_rest_headers,
            params={"assetType": "agents"},
        )
        items = response.get("items", [])
        source_ids = {item["id"] for item in items}

        assert configured_agent_catalog_ids <= source_ids, (
            f"Expected agent sources {configured_agent_catalog_ids} in response, got {source_ids}"
        )

        available_sources = [item for item in items if item.get("status") == "available"]
        assert available_sources, f"Expected at least one available agent source, got: {items}"
        LOGGER.info(f"Agent sources returned: {source_ids}")
