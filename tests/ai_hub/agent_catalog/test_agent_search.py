from typing import Self

import pytest

from tests.ai_hub.agent_catalog.constants import LANGGRAPH_FRAMEWORK
from tests.ai_hub.agent_catalog.utils import (
    assert_paginated_agents_unique_and_filtered,
    paginate_filtered_agents,
)
from tests.ai_hub.utils import execute_get_command_with_retry

pytestmark = [
    pytest.mark.tier1,
    pytest.mark.usefixtures(
        "updated_dsc_component_state_scope_session",
        "model_registry_namespace",
        "agent_catalog_configmap_patch",
    ),
]


class TestAgentCatalogSearch:
    """Tests for agent catalog filterQuery and pagination (RHOAIENG-70683)."""

    def test_filter_by_framework_name(
        self: Self,
        agent_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        langgraph_framework_filter_query: str,
        expected_langgraph_agent_names: set[str],
    ) -> None:
        """Given agents with a known framework exist in the catalog
        When filtering with filterQuery=framework='langgraph'
        Then only matching agents are returned
        """
        response = execute_get_command_with_retry(
            url=f"{agent_catalog_rest_urls[0]}agents",
            headers=model_registry_rest_headers,
            params={"filterQuery": langgraph_framework_filter_query},
        )
        items = response.get("items", [])
        returned_names = {item["name"] for item in items}

        assert returned_names == expected_langgraph_agent_names, (
            f"Expected agents {expected_langgraph_agent_names} for filter "
            f"'{langgraph_framework_filter_query}', got {returned_names}"
        )
        for item in items:
            assert item["framework"] == LANGGRAPH_FRAMEWORK, (
                f"Agent '{item['name']}' has framework='{item.get('framework')}', expected '{LANGGRAPH_FRAMEWORK}'"
            )

    @pytest.mark.parametrize(
        "order_params",
        [
            pytest.param({}, id="test_without_order_by"),
            pytest.param({"orderBy": "id", "sortOrder": "ASC"}, id="test_with_order_by_asc"),
            pytest.param({"orderBy": "id", "sortOrder": "DESC"}, id="test_with_order_by_desc"),
        ],
    )
    def test_pagination_with_filters(
        self: Self,
        agent_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        langgraph_framework_filter_query: str,
        expected_langgraph_agent_names: set[str],
        order_params: dict[str, str],
    ) -> None:
        """Given multiple agents share the same framework
        When paginating filtered results with pageSize=1
        Then each page returns a unique agent and all pages cover the filtered set
        """
        base_url = f"{agent_catalog_rest_urls[0]}agents"
        items, total_items = paginate_filtered_agents(
            base_url=base_url,
            headers=model_registry_rest_headers,
            filter_query=langgraph_framework_filter_query,
            page_size=1,
            order_params=order_params,
        )

        assert total_items == len(expected_langgraph_agent_names), (
            f"Expected {len(expected_langgraph_agent_names)} agents matching filter "
            f"'{langgraph_framework_filter_query}', got {total_items}"
        )
        assert_paginated_agents_unique_and_filtered(
            items=items,
            total_items=total_items,
            order_params=order_params,
            field_name="framework",
            expected_field_value=LANGGRAPH_FRAMEWORK,
        )

        returned_names = {item["name"] for item in items}
        assert returned_names == expected_langgraph_agent_names, (
            f"Expected agents {expected_langgraph_agent_names} for filter "
            f"'{langgraph_framework_filter_query}', got {returned_names}"
        )
