from typing import Self

import pytest
import structlog

from tests.ai_hub.agent_catalog.utils import (
    assert_paginated_agents_unique_and_filtered,
    paginate_filtered_agents,
)
from tests.ai_hub.utils import execute_get_command_with_retry

LOGGER = structlog.get_logger(name=__name__)


class TestAgentCatalogPagination:
    """Tests for agent catalog pagination and ordering (RHOAIENG-70683)."""

    def test_page_size_limits_results(
        self: Self,
        agent_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
    ) -> None:
        """Given more agents exist than the requested page size
        When listing with pageSize=2
        Then at most 2 agents are returned with a nextPageToken
        """
        page_size = 2
        response = execute_get_command_with_retry(
            url=f"{agent_catalog_rest_urls[0]}agents",
            headers=model_registry_rest_headers,
            params={"pageSize": str(page_size)},
        )
        items = response.get("items", [])
        assert len(items) <= page_size, f"Expected at most {page_size} agents, got {len(items)}"
        total_size = response.get("size", 0)
        if total_size > page_size:
            assert response.get("nextPageToken"), "Expected nextPageToken when more results exist"

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
        order_params: dict[str, str],
    ) -> None:
        """Given multiple agents share the same framework
        When paginating filtered results with pageSize=1
        Then each page returns a unique agent and all pages cover the filtered set
        """
        filter_query = "framework='langgraph'"
        base_url = f"{agent_catalog_rest_urls[0]}agents"
        items, total_items = paginate_filtered_agents(
            base_url=base_url,
            headers=model_registry_rest_headers,
            filter_query=filter_query,
            order_params=order_params,
        )

        assert total_items > 0, "Expected at least one langgraph agent for pagination test"
        assert_paginated_agents_unique_and_filtered(
            items=items,
            total_items=total_items,
            order_params=order_params,
            field_name="framework",
            expected_field_value="langgraph",
        )

    @pytest.mark.tier2
    @pytest.mark.parametrize(
        "sort_order",
        [
            pytest.param("ASC", id="test_order_by_name_asc"),
            pytest.param("DESC", id="test_order_by_name_desc"),
        ],
    )
    def test_order_by_name(
        self: Self,
        agent_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        sort_order: str,
    ) -> None:
        """Given multiple agents exist in the catalog
        When listing with orderBy=name and sortOrder
        Then agents are sorted alphabetically by name
        """
        response = execute_get_command_with_retry(
            url=f"{agent_catalog_rest_urls[0]}agents",
            headers=model_registry_rest_headers,
            params={"orderBy": "name", "sortOrder": sort_order, "pageSize": "1000"},
        )
        names = [item["name"] for item in response.get("items", [])]
        expected = sorted(names) if sort_order == "ASC" else sorted(names, reverse=True)
        assert names == expected, f"Expected names in {sort_order} order: {expected}, got {names}"

    @pytest.mark.tier2
    @pytest.mark.parametrize(
        "sort_order",
        [
            pytest.param("ASC", id="test_order_by_create_time_asc"),
            pytest.param("DESC", id="test_order_by_create_time_desc"),
        ],
    )
    def test_order_by_create_time(
        self: Self,
        agent_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        sort_order: str,
    ) -> None:
        """Given agents with different creation timestamps exist
        When listing with orderBy=createTimeSinceEpoch and sortOrder
        Then agents are sorted by creation time
        """
        response = execute_get_command_with_retry(
            url=f"{agent_catalog_rest_urls[0]}agents",
            headers=model_registry_rest_headers,
            params={"orderBy": "createTimeSinceEpoch", "sortOrder": sort_order, "pageSize": "1000"},
        )
        timestamps = [item["createTimeSinceEpoch"] for item in response.get("items", [])]
        expected = sorted(timestamps) if sort_order == "ASC" else sorted(timestamps, reverse=True)
        assert timestamps == expected, f"Expected timestamps in {sort_order} order: {expected}, got {timestamps}"
