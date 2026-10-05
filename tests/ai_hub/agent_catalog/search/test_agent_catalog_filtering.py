from typing import Any, Self

import pytest
import structlog

from tests.ai_hub.agent_catalog.config.constants import DEFAULT_AGENT_SOURCE_LABEL
from tests.ai_hub.agent_catalog.constants import (
    AUTOGEN_FRAMEWORK,
    CLAUDE_CODE_FRAMEWORK,
    CREWAI_FRAMEWORK,
    LANGGRAPH_FRAMEWORK,
)
from tests.ai_hub.agent_catalog.utils import filter_agents_match_criteria_or
from tests.ai_hub.utils import execute_get_command_with_retry

LOGGER = structlog.get_logger(name=__name__)


class TestAgentCatalogFiltering:
    """Tests for agent catalog filterQuery operators (RHOAIENG-70683)."""

    @pytest.mark.parametrize(
        "params, criteria_groups",
        [
            pytest.param(
                {"filterQuery": f"framework='{LANGGRAPH_FRAMEWORK}'"},
                [[{"field": "framework", "operator": "=", "value": LANGGRAPH_FRAMEWORK}]],
                id="test_filter_equals",
            ),
            pytest.param(
                {"filterQuery": "displayName LIKE 'LangGraph%'"},
                [[{"field": "displayName", "operator": "LIKE", "value": "LangGraph%"}]],
                id="test_filter_like",
            ),
            pytest.param(
                {"filterQuery": "name ILIKE '%REACT%'"},
                [[{"field": "name", "operator": "ILIKE", "value": "%REACT%"}]],
                id="test_filter_ilike",
            ),
            pytest.param(
                {"filterQuery": f"framework IN ('{LANGGRAPH_FRAMEWORK}','{CREWAI_FRAMEWORK}')"},
                [[{"field": "framework", "operator": "IN", "value": [LANGGRAPH_FRAMEWORK, CREWAI_FRAMEWORK]}]],
                id="test_filter_in",
            ),
            pytest.param(
                {"filterQuery": f"framework='{LANGGRAPH_FRAMEWORK}' AND name LIKE '%rag%'"},
                [
                    [
                        {"field": "framework", "operator": "=", "value": LANGGRAPH_FRAMEWORK},
                        {"field": "name", "operator": "LIKE", "value": "%rag%"},
                    ]
                ],
                id="test_filter_and",
            ),
            pytest.param(
                {"filterQuery": f"framework='{AUTOGEN_FRAMEWORK}' OR framework='{CLAUDE_CODE_FRAMEWORK}'"},
                [
                    [{"field": "framework", "operator": "=", "value": AUTOGEN_FRAMEWORK}],
                    [{"field": "framework", "operator": "=", "value": CLAUDE_CODE_FRAMEWORK}],
                ],
                id="test_filter_or",
            ),
            pytest.param(
                {
                    "filterQuery": (
                        f"(framework='{LANGGRAPH_FRAMEWORK}' OR framework='{CREWAI_FRAMEWORK}') AND name LIKE '%agent%'"
                    )
                },
                [
                    [
                        {"field": "framework", "operator": "=", "value": LANGGRAPH_FRAMEWORK},
                        {"field": "name", "operator": "LIKE", "value": "%agent%"},
                    ],
                    [
                        {"field": "framework", "operator": "=", "value": CREWAI_FRAMEWORK},
                        {"field": "name", "operator": "LIKE", "value": "%agent%"},
                    ],
                ],
                id="test_filter_nested_and_or",
                marks=pytest.mark.tier2,
            ),
        ],
    )
    def test_filter_returns_matching_agents(
        self: Self,
        agent_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        default_agents: list[dict[str, Any]],
        params: dict[str, str],
        criteria_groups: list[list[dict[str, str]]],
    ) -> None:
        """Given agents exist in the catalog
        When filtering with the given filterQuery
        Then the API returns exactly the agents matching the criteria
        """
        response = execute_get_command_with_retry(
            url=f"{agent_catalog_rest_urls[0]}agents",
            headers=model_registry_rest_headers,
            params=params,
        )
        returned_names = {item["name"] for item in response.get("items", [])}
        LOGGER.info(f"Returned {returned_names} agents")
        expected_names = filter_agents_match_criteria_or(agents=default_agents, criteria_groups=criteria_groups)
        LOGGER.info(f"Expected {expected_names} agents")
        assert returned_names == expected_names, (
            f"Missing: {expected_names - returned_names}, Unexpected: {returned_names - expected_names}"
        )

    def test_filter_not_equals(
        self: Self,
        agent_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
    ) -> None:
        """Given agents with different frameworks exist
        When filtering with filterQuery=framework!='crewai'
        Then no returned agent has framework crewai
        """
        response = execute_get_command_with_retry(
            url=f"{agent_catalog_rest_urls[0]}agents",
            headers=model_registry_rest_headers,
            params={"filterQuery": f"framework!='{CREWAI_FRAMEWORK}'"},
        )
        items = response.get("items", [])
        assert items, "Expected at least one agent after excluding crewai"
        for item in items:
            assert item.get("framework") != CREWAI_FRAMEWORK, (
                f"Agent '{item['name']}' has framework '{CREWAI_FRAMEWORK}' but should be excluded"
            )

    def test_filter_valid_source_label(
        self: Self,
        agent_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        default_agents: list[dict[str, Any]],
    ) -> None:
        """Given agents are configured with a default source label
        When filtering by a valid sourceLabel
        Then only agents from that source are returned
        """
        response = execute_get_command_with_retry(
            url=f"{agent_catalog_rest_urls[0]}agents",
            headers=model_registry_rest_headers,
            params={"sourceLabel": DEFAULT_AGENT_SOURCE_LABEL, "pageSize": "1000"},
        )
        filtered_names = {item["name"] for item in response.get("items", [])}
        expected_names = {agent["name"] for agent in default_agents if agent.get("source_id") == "rh_agents"}
        assert filtered_names == expected_names, (
            f"Missing: {expected_names - filtered_names}, Unexpected: {filtered_names - expected_names}"
        )

    def test_filter_invalid_source_label(
        self: Self,
        agent_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
    ) -> None:
        """Given no agents are configured with the given source label
        When filtering by an invalid sourceLabel
        Then an empty result set is returned
        """
        response = execute_get_command_with_retry(
            url=f"{agent_catalog_rest_urls[0]}agents",
            headers=model_registry_rest_headers,
            params={"sourceLabel": "nonexistent"},
        )
        assert response.get("size", 0) == 0, "Expected 0 agents for invalid sourceLabel"

    def test_filter_agents_by_label(
        self: Self,
        agent_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
    ) -> None:
        """Given agents with labels exist in the catalog
        When filtering by a specific label
        Then only agents with that label are returned
        """
        response = execute_get_command_with_retry(
            url=f"{agent_catalog_rest_urls[0]}agents",
            headers=model_registry_rest_headers,
            params={"filterQuery": "labels='rag'", "pageSize": 1000},
        )
        items = response.get("items", [])
        assert items, "Expected at least one agent with label 'rag'"
        for item in items:
            assert "rag" in item.get("labels", []), (
                f"Agent '{item['name']}' returned by rag filter but doesn't have 'rag' label"
            )
