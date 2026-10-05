from tests.ai_hub.utils import execute_get_command_with_retry


def get_agent_id_by_name(
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
