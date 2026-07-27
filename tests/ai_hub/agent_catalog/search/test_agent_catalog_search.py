from typing import Any, Self

import pytest
import structlog

from tests.ai_hub.agent_catalog.constants import LANGGRAPH_FRAMEWORK
from tests.ai_hub.agent_catalog.utils import search_agents
from tests.ai_hub.utils import execute_get_command_with_retry

LOGGER = structlog.get_logger(name=__name__)


class TestAgentCatalogSearch:
    """Tests for agent catalog keyword search and name matching (RHOAIENG-70683)."""

    @pytest.mark.parametrize(
        "params, criteria",
        [
            pytest.param(
                {"q": "ReAct"},
                None,
                id="test_keyword_search",
            ),
            pytest.param(
                {"name": "%langgraph%"},
                [{"field": "name", "operator": "LIKE", "value": "%langgraph%"}],
                id="test_name_like_search",
            ),
            pytest.param(
                {"q": "agent", "filterQuery": f"framework='{LANGGRAPH_FRAMEWORK}'"},
                [{"field": "framework", "operator": "=", "value": LANGGRAPH_FRAMEWORK}],
                id="test_combined_search_and_filter",
            ),
        ],
    )
    def test_search_returns_matching_agents(
        self: Self,
        agent_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        default_agents: list[dict[str, Any]],
        params: dict[str, str],
        criteria: list[dict[str, str]] | None,
    ) -> None:
        """Given agents exist in the catalog
        When searching with the given parameters
        Then the API returns exactly the agents matching the search criteria
        """
        response = execute_get_command_with_retry(
            url=f"{agent_catalog_rest_urls[0]}agents",
            headers=model_registry_rest_headers,
            params=params,
        )
        returned_names = {item["name"] for item in response.get("items", [])}
        LOGGER.info(f"Returned agents names: {returned_names}")
        search_params = {key: val for key, val in params.items() if key in ("q", "name")}
        expected_names = search_agents(agents=default_agents, params=search_params, criteria=criteria)
        LOGGER.info(f"Expected agents names: {expected_names}")
        assert returned_names == expected_names, (
            f"Missing: {expected_names - returned_names}, Unexpected: {returned_names - expected_names}"
        )

    def test_keyword_search_no_matches(
        self: Self,
        agent_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
    ) -> None:
        """Given no agents match the search term
        When searching with q=nonexistentagentxyz123
        Then an empty result set is returned
        """
        response = execute_get_command_with_retry(
            url=f"{agent_catalog_rest_urls[0]}agents",
            headers=model_registry_rest_headers,
            params={"q": "nonexistentagentxyz123"},
        )
        assert response.get("size", 0) == 0, (
            f"Expected 0 results for nonexistent search term, got {response.get('size', 0)}"
        )
