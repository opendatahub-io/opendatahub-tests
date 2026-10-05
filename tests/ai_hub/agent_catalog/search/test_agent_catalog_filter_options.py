from typing import Self

import pytest
import structlog

from tests.ai_hub.agent_catalog.search.constants import EXPECTED_FILTER_OPTIONS
from tests.ai_hub.utils import execute_get_command_with_retry

LOGGER = structlog.get_logger(name=__name__)

pytestmark = [
    pytest.mark.tier1,
    pytest.mark.usefixtures(
        "updated_dsc_component_state_scope_session",
        "model_registry_namespace",
    ),
]


class TestAgentCatalogFilterOptions:
    """Tests for agent catalog filter_options endpoint (RHOAIENG-70683)."""

    def test_filter_options_returns_expected_fields(
        self: Self,
        agent_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
    ) -> None:
        """Given the agent catalog is running
        When requesting GET /agents/filter_options
        Then filterable fields for filterQuery are returned
        """
        url = f"{agent_catalog_rest_urls[0]}agents/filter_options"
        LOGGER.info(f"Requesting filter_options from: {url}")

        response = execute_get_command_with_retry(
            url=url,
            headers=model_registry_rest_headers,
        )
        LOGGER.info(f"filter_options response keys: {list(response.get('filters', {}).keys())}")

        assert EXPECTED_FILTER_OPTIONS <= set(response["filters"].keys()), (
            f"Expected filter options {EXPECTED_FILTER_OPTIONS} not found in {set(response['filters'].keys())}"
        )
