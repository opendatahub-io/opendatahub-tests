from typing import Any

import structlog
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap

from tests.ai_hub.constants import DEFAULT_CUSTOM_MODEL_CATALOG
from tests.ai_hub.utils import execute_get_command_with_retry

LOGGER = structlog.get_logger(name=__name__)


def get_agent_catalog_sources(admin_client: DynamicClient, model_registry_namespace: str) -> tuple[ConfigMap, dict]:
    """Return the user-editable catalog sources ConfigMap and its parsed sources.yaml data."""
    catalog_config_map = ConfigMap(
        name=DEFAULT_CUSTOM_MODEL_CATALOG,
        client=admin_client,
        namespace=model_registry_namespace,
    )
    current_data = yaml.safe_load(catalog_config_map.instance.data.get("sources.yaml", "{}") or "{}")
    return catalog_config_map, current_data


def _evaluate_agent_criterion(agent: dict[str, Any], criterion: dict[str, str]) -> tuple[bool, str]:
    """Evaluate a single filter criterion against an agent.

    Args:
        agent: Agent dictionary from API response.
        criterion: Filter criterion with ``field``, ``operator``, and ``value`` keys.

    Returns:
        Tuple of (condition_met, descriptive message).
    """
    field = criterion["field"]
    operator = criterion["operator"]
    expected = criterion["value"]
    agent_name = agent.get("name", "<unknown>")
    actual = agent.get(field)

    if actual is None:
        return False, f"Agent '{agent_name}' missing field '{field}'"

    condition_met = False
    if operator == "=":
        condition_met = actual == expected
    elif operator in ("!=", "<>"):
        condition_met = actual != expected
    elif operator == "LIKE":
        pattern = expected.replace("%", "")
        if expected.startswith("%") and expected.endswith("%"):
            condition_met = pattern in actual
        elif expected.startswith("%"):
            condition_met = actual.endswith(pattern)
        elif expected.endswith("%"):
            condition_met = actual.startswith(pattern)
        else:
            condition_met = actual == expected
    elif operator == "ILIKE":
        pattern = expected.replace("%", "").lower()
        actual_lower = actual.lower()
        if expected.startswith("%") and expected.endswith("%"):
            condition_met = pattern in actual_lower
        elif expected.startswith("%"):
            condition_met = actual_lower.endswith(pattern)
        elif expected.endswith("%"):
            condition_met = actual_lower.startswith(pattern)
        else:
            condition_met = actual_lower == pattern
    elif operator == "IN":
        condition_met = actual in expected

    message = f"Agent '{agent_name}' {field}: '{actual}' {operator} '{expected}'"
    return condition_met, message


def _get_agent_validation_results(
    agent: dict[str, Any], criteria: list[dict[str, str]]
) -> tuple[list[bool], list[str]]:
    """Evaluate all criteria against a single agent.

    Args:
        agent: Agent dictionary from API response.
        criteria: List of filter criteria dicts.

    Returns:
        Tuple of (boolean results list, message list).
    """
    bool_results = []
    messages = []
    for criterion in criteria:
        condition_met, message = _evaluate_agent_criterion(agent=agent, criterion=criterion)
        bool_results.append(condition_met)
        messages.append(message)
    return bool_results, messages


def evaluate_agent_search(agent: dict[str, Any], params: dict[str, str]) -> bool:
    """Evaluate whether an agent matches the given search parameters.

    Supports ``q`` (keyword across searchable fields) and ``name`` (LIKE match).
    When both ``q`` and ``filterQuery`` criteria are provided, the caller should
    combine this with ``filter_agents_match_criteria_and``.

    Args:
        agent: Agent dictionary from API response.
        params: Search parameters (``q`` and/or ``name``).

    Returns:
        True if the agent matches all provided search parameters.
    """
    if "q" in params:
        keyword = params["q"].lower()
        searchable_fields = [
            agent.get("name", "").lower(),
            agent.get("displayName", "").lower(),
            agent.get("description", "").lower(),
        ]
        if not any(keyword in content for content in searchable_fields if content):
            return False

    if "name" in params:
        name_param = params["name"]
        criterion = {"field": "name", "operator": "LIKE", "value": name_param}
        matched, _ = _evaluate_agent_criterion(agent=agent, criterion=criterion)
        if not matched:
            return False

    return True


def search_agents(
    agents: list[dict[str, Any]],
    params: dict[str, str],
    criteria: list[dict[str, str]] | None = None,
) -> set[str]:
    """Return names of agents matching search params and optional filter criteria.

    Args:
        agents: Full list of agents from API.
        params: Search parameters (``q`` and/or ``name``).
        criteria: Optional filter criteria applied with AND logic.

    Returns:
        Set of agent names matching all conditions.
    """
    matched = set()
    for agent in agents:
        if not evaluate_agent_search(agent=agent, params=params):
            continue
        if criteria:
            bool_results, _ = _get_agent_validation_results(agent=agent, criteria=criteria)
            if not all(bool_results):
                continue
        matched.add(agent["name"])
    return matched


def filter_agents_match_criteria_and(agents: list[dict[str, Any]], criteria: list[dict[str, str]]) -> set[str]:
    """Return names of agents satisfying ALL criteria.

    Args:
        agents: Full list of agents from API.
        criteria: List of filter criteria dicts with ``field``, ``operator``, ``value``.

    Returns:
        Set of agent names matching all criteria.
    """
    matched = set()
    for agent in agents:
        bool_results, messages = _get_agent_validation_results(agent=agent, criteria=criteria)
        if all(bool_results):
            matched.add(agent["name"])
            LOGGER.info(f"Matched (AND): {[f'{msg}: passed' for msg in messages]}")
    return matched


def filter_agents_match_criteria_or(
    agents: list[dict[str, Any]], criteria_groups: list[list[dict[str, str]]]
) -> set[str]:
    """Return names of agents satisfying ANY criteria group (each group uses AND internally).

    Args:
        agents: Full list of agents from API.
        criteria_groups: List of criteria groups; agent matches if any group fully matches.

    Returns:
        Set of agent names matching at least one criteria group.
    """
    matched = set()
    for agent in agents:
        for group in criteria_groups:
            bool_results, messages = _get_agent_validation_results(agent=agent, criteria=group)
            if all(bool_results):
                matched.add(agent["name"])
                LOGGER.info(f"Matched (OR): {messages[bool_results.index(True)]}")
                break
    return matched


def paginate_filtered_agents(
    base_url: str,
    headers: dict[str, str],
    filter_query: str,
    order_params: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Paginate through filtered agent catalog results one item at a time.

    Fetches the total count, then walks each page with pageSize=1 and
    optional ordering parameters.

    Args:
        base_url: Agents list endpoint URL (e.g. ``.../agents``).
        headers: REST request headers.
        filter_query: filterQuery expression applied to every page request.
        order_params: Optional ordering parameters (orderBy, sortOrder).

    Returns:
        Tuple of (items collected across all pages, total count from API).

    Raises:
        AssertionError: If any page returns an unexpected item count or missing nextPageToken.
    """
    order_params = order_params or {}

    all_response = execute_get_command_with_retry(
        url=base_url,
        headers=headers,
        params={"filterQuery": filter_query},
    )
    total_items = all_response.get("size", 0)
    LOGGER.info(f"Total items matching filter '{filter_query}': {total_items}")

    items: list[dict[str, Any]] = []
    next_page_token: str | None = None

    for page_num in range(1, total_items + 1):
        params: dict[str, str] = {
            "filterQuery": filter_query,
            "pageSize": "1",
            **order_params,
        }
        if next_page_token:
            params["nextPageToken"] = next_page_token

        response = execute_get_command_with_retry(
            url=base_url,
            headers=headers,
            params=params,
        )
        page_items = response.get("items", [])
        assert len(page_items) == 1, f"Expected 1 item on page {page_num}, got {len(page_items)}"
        items.extend(page_items)

        next_page_token = response.get("nextPageToken")
        if page_num < total_items:
            assert next_page_token, f"Expected nextPageToken after page {page_num}, but got none"

    LOGGER.info(f"Pagination complete: collected {len(items)} agents out of {total_items} total")
    return items, total_items


def assert_paginated_agents_unique_and_filtered(
    items: list[dict[str, Any]],
    total_items: int,
    order_params: dict[str, str],
    field_name: str,
    expected_field_value: str,
) -> None:
    """Assert paginated agents are unique, correctly ordered, and match the filter field.

    Args:
        items: Agent items collected across all pagination pages.
        total_items: Expected total item count from the API.
        order_params: Ordering parameters used during pagination (orderBy, sortOrder).
        field_name: Item field that must match the filter (e.g. ``framework``).
        expected_field_value: Expected value for field_name on every item.

    Raises:
        AssertionError: If uniqueness, ordering, count, or field validation fails.
    """
    seen_ids: list[str] = []
    for page_idx, item in enumerate(items, start=1):
        assert item[field_name] == expected_field_value, (
            f"Page {page_idx} agent id '{item['id']}' has {field_name}='{item.get(field_name)}',"
            f" expected '{expected_field_value}'"
        )

        agent_id = item["id"]
        assert agent_id not in seen_ids, (
            f"Page {page_idx} returned duplicate agent id '{agent_id}', already seen on a previous page"
        )

        sort_order = order_params.get("sortOrder", "ASC")
        if seen_ids:
            if sort_order == "ASC":
                assert int(agent_id) > int(seen_ids[-1]), (
                    f"Page {page_idx} id '{agent_id}' is not greater than previous id '{seen_ids[-1]}'"
                )
            elif sort_order == "DESC":
                assert int(agent_id) < int(seen_ids[-1]), (
                    f"Page {page_idx} id '{agent_id}' is not less than previous id '{seen_ids[-1]}'"
                )
        seen_ids.append(agent_id)

    assert len(seen_ids) == total_items, f"Pagination returned {len(seen_ids)} unique agents but expected {total_items}"
