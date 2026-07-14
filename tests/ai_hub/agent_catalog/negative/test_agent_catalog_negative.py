from typing import Any, Self

import pytest
import structlog
from kubernetes.dynamic.exceptions import ResourceNotFoundError

from tests.ai_hub.utils import execute_get_command

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.tier3
class TestAgentCatalogNegative:
    """Negative and error handling tests for agent catalog API (RHOAIENG-70683)."""

    @pytest.mark.parametrize(
        "path, params",
        [
            pytest.param(
                "agents/999999",
                None,
                id="test_nonexistent_agent_id",
            ),
            pytest.param(
                "sources/nonexistent-source/agents/langgraph-react-agent",
                None,
                id="test_nonexistent_source_id",
            ),
            pytest.param(
                "sources/rh_agents/agents/nonexistent-agent",
                None,
                id="test_nonexistent_agent_name",
            ),
            pytest.param(
                "agents",
                {"filterQuery": "INVALID SYNTAX !!!"},
                id="test_invalid_filter_query",
            ),
            pytest.param(
                "agents",
                {"nextPageToken": "bogus_token"},
                id="test_invalid_next_page_token",
                marks=pytest.mark.tier2,
            ),
            pytest.param(
                "agents/999999/artifacts",
                None,
                id="test_artifacts_nonexistent_agent",
            ),
        ],
    )
    def test_invalid_request_returns_error(
        self: Self,
        agent_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        path: str,
        params: dict[str, str] | None,
    ) -> None:
        """Given an invalid request (nonexistent resource or bad syntax)
        When sending the request to the agent catalog API
        Then an error is returned
        """
        with pytest.raises(ResourceNotFoundError):
            execute_get_command(
                url=f"{agent_catalog_rest_urls[0]}{path}",
                headers=model_registry_rest_headers,
                params=params,
            )

    def test_artifacts_invalid_artifact_type(
        self: Self,
        agent_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        default_agents: list[dict[str, Any]],
    ) -> None:
        """Given a valid agent ID
        When requesting artifacts with artifactType=image-artifact
        Then a 400 error is returned
        """
        agent_id = default_agents[0]["id"]
        with pytest.raises(ResourceNotFoundError):
            execute_get_command(
                url=f"{agent_catalog_rest_urls[0]}agents/{agent_id}/artifacts",
                headers=model_registry_rest_headers,
                params={"artifactType": "image-artifact"},
            )
