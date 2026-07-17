from typing import Any, Self

import pytest
import structlog
from dictdiffer import diff

from tests.ai_hub.utils import execute_get_command_with_retry

LOGGER = structlog.get_logger(name=__name__)

REQUIRED_API_AGENT_FIELDS: set[str] = {"id", "name", "description", "displayName", "source_id"}

# Fields added by the API that are not in the source YAML
API_ONLY_FIELDS: set[str] = {"id", "source_id", "createTimeSinceEpoch", "lastUpdateTimeSinceEpoch"}

# Fields present in the YAML but served via a separate API endpoint (artifacts)
YAML_ONLY_FIELDS: set[str] = {"templates"}

pytestmark = [
    pytest.mark.tier1,
    pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace"),
]


class TestDefaultAgentMetadata:
    """Tests for default agent catalog metadata validation (RHOAIENG-70682)."""

    def test_default_agents_have_required_fields(
        self: Self,
        default_agents: list[dict[str, Any]],
    ) -> None:
        """Given agents exist in the default catalog
        When inspecting each agent
        Then required fields are present and non-empty on every agent
        """
        errors: list[str] = []
        for agent in default_agents:
            agent_name = agent.get("name", "<unknown>")
            missing = REQUIRED_API_AGENT_FIELDS - set(agent.keys())
            if missing:
                errors.append(f"Agent '{agent_name}' missing required fields: {missing}")
            for field in REQUIRED_API_AGENT_FIELDS:
                if not agent.get(field):
                    errors.append(f"Agent '{agent_name}' has empty required field '{field}'")
        assert not errors, "Required field validation failed:\n" + "\n".join(errors)

    def test_yaml_agents_match_api_response(
        self: Self,
        default_agents: list[dict[str, Any]],
        default_agents_yaml_content: dict[str, Any],
    ) -> None:
        """Given agents are defined in the catalog YAML on the pod
        When comparing each YAML agent with the API response
        Then every field present in the YAML is returned by the API with the same value
        """
        api_agents = {agent["name"]: agent for agent in default_agents}
        agents_with_differences: dict[str, list] = {}

        for yaml_agent in default_agents_yaml_content.get("agents", []):
            agent_name = yaml_agent["name"]
            LOGGER.info(f"Validating agent: {agent_name}")

            api_agent = api_agents.get(agent_name)
            assert api_agent, f"Agent '{agent_name}' found in YAML but not in API response"

            yaml_fields = {
                key: value for key, value in yaml_agent.items() if key not in API_ONLY_FIELDS | YAML_ONLY_FIELDS
            }
            api_fields = {key: value for key, value in api_agent.items() if key in yaml_fields}

            differences = list(diff(yaml_fields, api_fields))
            if differences:
                agents_with_differences[agent_name] = differences
                LOGGER.warning(f"Found differences for '{agent_name}': {differences}")

        assert not agents_with_differences, (
            f"Found differences in {len(agents_with_differences)} agent(s): {agents_with_differences}"
        )

    def test_agents_with_repository_url_have_readme(
        self: Self,
        default_agents: list[dict[str, Any]],
    ) -> None:
        """Given agents with repositoryUrl exist in the default catalog
        When inspecting each agent
        Then agents with repositoryUrl also have a non-empty readme
        """
        errors: list[str] = []
        for agent in default_agents:
            agent_name = agent.get("name", "<unknown>")
            if agent.get("repositoryUrl") and not agent.get("readme"):
                errors.append(f"Agent '{agent_name}' has repositoryUrl but missing readme")
        assert not errors, "README auto-population failed:\n" + "\n".join(errors)

    def test_agent_get_by_id_matches_list(
        self: Self,
        agent_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        default_agents: list[dict[str, Any]],
    ) -> None:
        """Given agents exist in the default catalog
        When fetching an agent by ID
        Then the response matches the corresponding entry from the list endpoint
        """
        agent = default_agents[0]
        fetched = execute_get_command_with_retry(
            url=f"{agent_catalog_rest_urls[0]}agents/{agent['id']}",
            headers=model_registry_rest_headers,
        )
        LOGGER.info(f"Comparing list vs get for agent '{agent['name']}'")
        assert fetched == agent, (
            f"Agent fetched by ID does not match list response.\nExpected: {agent}\nActual: {fetched}"
        )
