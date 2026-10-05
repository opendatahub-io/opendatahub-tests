from typing import Self

import pytest
import structlog

from tests.ai_hub.agent_catalog.artifacts.constants import (
    ARTIFACT_IMAGE_COUNT,
    ARTIFACT_IMAGE_ONLY_COUNT,
    ARTIFACT_TEMPLATE_COUNT,
    ARTIFACT_TEST_AGENTS_YAML,
    ARTIFACT_TEST_LABEL,
    ARTIFACT_TEST_LABEL_DEFINITION,
    ARTIFACT_TEST_SOURCE,
    DEFAULT_TEMPLATE_NAME,
    EXPECTED_TEMPLATE_NAMES,
    TEMPLATE_ARTIFACT_TYPE,
)
from tests.ai_hub.utils import execute_get_command_with_retry

LOGGER = structlog.get_logger(name=__name__)

ARTIFACT_CONFIGMAP_PARAM: dict = {
    "source": ARTIFACT_TEST_SOURCE,
    "label": ARTIFACT_TEST_LABEL,
    "label_definition": ARTIFACT_TEST_LABEL_DEFINITION,
    "agents_yaml": ARTIFACT_TEST_AGENTS_YAML,
    "min_agents": 4,
}


@pytest.mark.parametrize(
    "agent_catalog_configmap_patch",
    [pytest.param(ARTIFACT_CONFIGMAP_PARAM)],
    indirect=True,
)
class TestAgentArtifacts:
    """Tests for agent catalog artifacts endpoint (RHOAIENG-75429)."""

    @pytest.mark.parametrize(
        "artifact_type_filter",
        [
            pytest.param(None, id="test_list_all_artifacts"),
            pytest.param(TEMPLATE_ARTIFACT_TYPE, id="test_filter_by_template_type"),
        ],
    )
    def test_list_artifacts(
        self: Self,
        agent_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        artifact_agent_id: str,
        artifact_type_filter: str | None,
    ) -> None:
        """Given an agent with template artifacts exists
        When listing artifacts with or without artifactType filter
        Then template artifacts are returned with the expected count
        """
        params: dict[str, str] | None = {"artifactType": artifact_type_filter} if artifact_type_filter else None

        response = execute_get_command_with_retry(
            url=f"{agent_catalog_rest_urls[0]}agents/{artifact_agent_id}/artifacts",
            headers=model_registry_rest_headers,
            params=params,
        )
        items = response.get("items", [])
        LOGGER.info(f"Artifacts (filter={artifact_type_filter}): {len(items)} items")
        assert len(items) == ARTIFACT_TEMPLATE_COUNT, f"Expected {ARTIFACT_TEMPLATE_COUNT} artifacts, got {len(items)}"
        for item in items:
            assert item.get("artifactType") == TEMPLATE_ARTIFACT_TYPE, (
                f"Expected artifactType '{TEMPLATE_ARTIFACT_TYPE}', got '{item.get('artifactType')}'"
            )

    @pytest.mark.parametrize(
        "agent_fixture, expected_inline_count, expected_template_count",
        [
            pytest.param(
                "artifact_agent_id",
                ARTIFACT_IMAGE_COUNT,
                ARTIFACT_TEMPLATE_COUNT,
                id="test_full_agent",
            ),
            pytest.param(
                "empty_artifact_agent_id",
                0,
                0,
                id="test_empty_agent",
            ),
            pytest.param(
                "image_only_artifact_agent_id",
                ARTIFACT_IMAGE_ONLY_COUNT,
                0,
                id="test_image_only_agent",
            ),
        ],
    )
    def test_agent_artifact_types(
        self: Self,
        request: pytest.FixtureRequest,
        agent_catalog_configmap_patch: None,
        agent_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        agent_fixture: str,
        expected_inline_count: int,
        expected_template_count: int,
    ) -> None:
        """Given agents with different artifact configurations exist
        When fetching the agent and querying the artifacts endpoint
        Then inline image artifacts and template artifacts match expected counts
        """
        agent_id = request.getfixturevalue(argname=agent_fixture)

        agent = execute_get_command_with_retry(
            url=f"{agent_catalog_rest_urls[0]}agents/{agent_id}",
            headers=model_registry_rest_headers,
        )
        inline_artifacts = agent.get("artifacts", [])
        LOGGER.info(f"Agent {agent_id}: {len(inline_artifacts)} inline, expecting {expected_inline_count}")
        assert len(inline_artifacts) == expected_inline_count, (
            f"Expected {expected_inline_count} inline artifacts, got {len(inline_artifacts)}"
        )
        if inline_artifacts:
            for artifact in inline_artifacts:
                assert artifact.get("uri"), f"Inline artifact missing 'uri': {artifact}"

        response = execute_get_command_with_retry(
            url=f"{agent_catalog_rest_urls[0]}agents/{agent_id}/artifacts",
            headers=model_registry_rest_headers,
        )
        assert response.get("size", 0) == expected_template_count, (
            f"Expected {expected_template_count} template artifacts, got {response.get('size', 0)}"
        )

    def test_template_artifacts_have_content_and_names(
        self: Self,
        agent_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        artifact_agent_id: str,
    ) -> None:
        """Given an agent with named template artifacts exists
        When listing template artifacts
        Then each has a non-empty content field and the expected name
        """
        response = execute_get_command_with_retry(
            url=f"{agent_catalog_rest_urls[0]}agents/{artifact_agent_id}/artifacts",
            headers=model_registry_rest_headers,
        )
        items = response.get("items", [])
        returned_names = {item["name"] for item in items}
        assert returned_names == EXPECTED_TEMPLATE_NAMES, (
            f"Expected template names {EXPECTED_TEMPLATE_NAMES}, got {returned_names}"
        )
        for item in items:
            assert item.get("content"), f"Template '{item['name']}' has empty content"

    def test_unnamed_template_defaults_to_agent_yaml(
        self: Self,
        agent_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        default_name_artifact_agent_id: str,
    ) -> None:
        """Given an agent with a template that has no explicit name
        When listing its template artifacts
        Then the template name defaults to agent.yaml
        """
        response = execute_get_command_with_retry(
            url=f"{agent_catalog_rest_urls[0]}agents/{default_name_artifact_agent_id}/artifacts",
            headers=model_registry_rest_headers,
        )
        items = response.get("items", [])
        assert len(items) == 1, f"Expected 1 template artifact, got {len(items)}"
        assert items[0]["name"] == DEFAULT_TEMPLATE_NAME, (
            f"Expected default template name '{DEFAULT_TEMPLATE_NAME}', got '{items[0]['name']}'"
        )
        assert items[0].get("content"), "Default-named template has empty content"

    @pytest.mark.parametrize(
        "artifact_type_filter",
        [
            pytest.param(None, id="test_paginate_all"),
            pytest.param(TEMPLATE_ARTIFACT_TYPE, id="test_paginate_template_artifacts"),
        ],
    )
    def test_artifacts_pagination(
        self: Self,
        agent_catalog_rest_urls: list[str],
        model_registry_rest_headers: dict[str, str],
        artifact_agent_id: str,
        artifact_type_filter: str | None,
    ) -> None:
        """Given an agent with multiple artifacts exists
        When paginating through all pages with pageSize=1
        Then all artifacts are collected without duplicates
        """
        url = f"{agent_catalog_rest_urls[0]}agents/{artifact_agent_id}/artifacts"

        total_params: dict[str, str] | None = {"artifactType": artifact_type_filter} if artifact_type_filter else None
        total_response = execute_get_command_with_retry(
            url=url, headers=model_registry_rest_headers, params=total_params
        )
        total_count = total_response.get("size", 0)
        assert total_count > 1, f"Need more than 1 artifact to test pagination, got {total_count}"

        collected_ids: list[str] = []
        next_token: str | None = None

        for page_num in range(1, total_count + 1):
            params: dict[str, str] = {"pageSize": "1"}
            if artifact_type_filter:
                params["artifactType"] = artifact_type_filter
            if next_token:
                params["nextPageToken"] = next_token

            page = execute_get_command_with_retry(url=url, headers=model_registry_rest_headers, params=params)
            items = page.get("items", [])
            assert len(items) == 1, f"Expected 1 artifact on page {page_num}, got {len(items)}"

            artifact_id = items[0]["id"]
            assert artifact_id not in collected_ids, f"Duplicate artifact '{artifact_id}' on page {page_num}"
            if artifact_type_filter:
                assert items[0].get("artifactType") == artifact_type_filter, (
                    f"Page {page_num}: expected type '{artifact_type_filter}', got '{items[0].get('artifactType')}'"
                )
            collected_ids.append(artifact_id)

            next_token = page.get("nextPageToken")
            if page_num < total_count:
                assert next_token, f"Expected nextPageToken after page {page_num}"

        LOGGER.info(f"Pagination complete: collected {len(collected_ids)}/{total_count} artifacts")
        assert len(collected_ids) == total_count, f"Collected {len(collected_ids)} artifacts but expected {total_count}"
