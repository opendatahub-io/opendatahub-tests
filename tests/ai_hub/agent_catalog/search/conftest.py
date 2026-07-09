from typing import Any

import pytest

from tests.ai_hub.utils import execute_get_command_with_retry


@pytest.fixture(scope="class")
def all_agents(
    agent_catalog_rest_urls: list[str],
    model_registry_rest_headers: dict[str, str],
) -> list[dict[str, Any]]:
    """Fetch all agents from the catalog once per test class."""
    response = execute_get_command_with_retry(
        url=f"{agent_catalog_rest_urls[0]}agents",
        headers=model_registry_rest_headers,
        params={"pageSize": 1000},
    )
    items = response.get("items", [])
    assert items, "Expected at least one agent in the catalog"
    return items
