import pytest

from tests.model_registry.utils import execute_get_command
from utilities.opendatahub_logger import get_logger

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="class")
def mcp_servers_response(
    mcp_catalog_rest_urls: list[str],
    model_registry_rest_headers: dict[str, str],
) -> dict:
    """Class-scoped fixture that fetches the MCP servers list once per test class."""
    return execute_get_command(
        url=f"{mcp_catalog_rest_urls[0]}mcp_servers",
        headers=model_registry_rest_headers,
    )
