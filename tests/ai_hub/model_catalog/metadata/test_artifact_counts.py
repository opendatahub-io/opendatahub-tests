from typing import Any, Self

import pytest
import structlog

from tests.ai_hub.model_catalog.metadata.constants import (
    ALL_ARTIFACT_CATEGORIES,
    NO_ARTIFACT_CATALOG_ID,
    NO_ARTIFACT_CATALOG_YAML_FILENAME,
    NO_ARTIFACT_MODELS,
    NO_ARTIFACT_SOURCES_YAML,
    NO_ARTIFACT_YAML,
)
from tests.ai_hub.model_catalog.metadata.utils import get_artifact_counts_from_endpoint
from tests.ai_hub.model_catalog.utils import get_models_from_catalog_api
from tests.ai_hub.utils import execute_get_command

LOGGER = structlog.get_logger(name=__name__)

pytestmark = [
    pytest.mark.usefixtures(
        "updated_dsc_component_state_scope_session",
        "model_registry_namespace",
    )
]


@pytest.mark.tier2
@pytest.mark.parametrize(
    "model_detail_with_artifacts, expected_missing_categories",
    [
        pytest.param(
            ALL_ARTIFACT_CATEGORIES,
            ALL_ARTIFACT_CATEGORIES,
            id="test_all_categories",
        ),
        pytest.param(
            {"model-artifact"},
            {"model-artifact"},
            id="test_model_artifact_only",
        ),
        pytest.param(
            {"model-artifact", "performance-metrics"},
            {"model-artifact", "performance-metrics"},
            id="test_model_and_performance",
        ),
    ],
    indirect=True,
)
class TestArtifactCounts:
    """Tests for artifactCounts field on the model detail endpoint (RHOAIENG-62827)."""

    def test_model_detail_includes_artifact_counts(
        self: Self,
        model_detail_with_artifacts: dict[str, Any],
        expected_missing_categories: set[str],
    ):
        """
        Given a model with the required artifact categories,
        When calling the model detail endpoint,
        Then the response should include an artifactCounts field as a dict.
        """
        del expected_missing_categories  # required by class-level parametrize but unused here
        artifact_counts = model_detail_with_artifacts.get("artifactCounts")
        LOGGER.info(f"Model: {model_detail_with_artifacts['name']}, artifactCounts: {artifact_counts}")
        assert artifact_counts is not None, (
            f"artifactCounts missing from detail response for {model_detail_with_artifacts['name']}"
        )
        assert isinstance(artifact_counts, dict), f"artifactCounts should be a dict, got {type(artifact_counts)}"

    def test_artifact_counts_values_are_positive_integers(
        self: Self,
        model_detail_with_artifacts: dict[str, Any],
        expected_missing_categories: set[str],
    ):
        """
        Given a model's artifactCounts from the detail endpoint,
        When checking each category,
        Then keys should be strings and values should be positive integers.
        """
        del expected_missing_categories  # required by class-level parametrize but unused here
        artifact_counts = model_detail_with_artifacts.get("artifactCounts", {})
        LOGGER.info(f"Model: {model_detail_with_artifacts['name']}, artifactCounts: {artifact_counts}")
        for category, count in artifact_counts.items():
            assert isinstance(category, str), f"Key {category!r} should be a string"
            assert isinstance(count, int), f"Value for {category} should be an int, got {type(count).__name__}"
            assert count > 0, f"Value for {category} should be positive, got {count}"

    def test_missing_categories_absent_from_artifact_counts(
        self: Self,
        model_detail_with_artifacts: dict[str, Any],
        expected_missing_categories: set[str],
    ):
        """
        Given a model that lacks certain artifact categories,
        When checking artifactCounts from the detail endpoint,
        Then those categories should not appear in the response.
        """
        artifact_counts = model_detail_with_artifacts.get("artifactCounts", {})
        LOGGER.info(f"artifactCounts: {artifact_counts}, expected missing: {expected_missing_categories}")

        leaked = expected_missing_categories & set(artifact_counts.keys())
        assert not leaked, f"Categories expected to be absent but found in artifactCounts: {leaked}"

    def test_artifact_counts_match_artifacts_endpoint(
        self: Self,
        model_detail_with_artifacts: dict[str, Any],
        expected_missing_categories: set[str],
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        Given a model with artifacts,
        When comparing artifactCounts from the detail endpoint to actual artifact counts,
        Then the counts should match.
        """
        del expected_missing_categories  # required by class-level parametrize but unused here
        artifact_counts = model_detail_with_artifacts.get("artifactCounts", {})
        actual_counts = get_artifact_counts_from_endpoint(
            model_catalog_rest_url=model_catalog_rest_url[0],
            source_id=model_detail_with_artifacts["source_id"],
            model_name=model_detail_with_artifacts["name"],
            headers=model_registry_rest_headers,
        )
        LOGGER.info(f"Detail: {artifact_counts}, Artifacts endpoint: {actual_counts}")

        errors = []
        for category, expected_count in artifact_counts.items():
            actual = actual_counts.get(category, 0)
            if actual != expected_count:
                errors.append(f"{category}: detail={expected_count}, artifacts endpoint={actual}")
        for category, count in actual_counts.items():
            if category not in artifact_counts:
                errors.append(f"{category}: missing from artifactCounts but found {count} in artifacts")

        assert not errors, f"Mismatch for {model_detail_with_artifacts['name']}:\n" + "\n".join(
            f"  - {err}" for err in errors
        )

    def test_model_list_excludes_artifact_counts(
        self: Self,
        model_detail_with_artifacts: dict[str, Any],
        expected_missing_categories: set[str],
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        Given a model with known artifactCounts on the detail endpoint,
        When fetching all models from the list endpoint for that source,
        Then no model should contain an artifactCounts field.
        """
        del expected_missing_categories  # required by class-level parametrize but unused here
        source_id = model_detail_with_artifacts["source_id"]

        response = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            page_size=1000,
            additional_params=f"&source={source_id}",
        )
        items = response.get("items", [])
        assert items, f"No models found in list endpoint for source {source_id}"

        models_with_counts = [item["name"] for item in items if "artifactCounts" in item]
        assert not models_with_counts, (
            f"artifactCounts should not appear on list endpoint, found on: {models_with_counts}"
        )
        LOGGER.info(f"Confirmed: {len(items)} models from source {source_id} have no artifactCounts on list endpoint")


@pytest.mark.tier2
@pytest.mark.parametrize(
    "updated_catalog_config_map",
    [
        pytest.param(
            {
                "sources_yaml": NO_ARTIFACT_SOURCES_YAML,
                "sample_yaml": {
                    NO_ARTIFACT_CATALOG_YAML_FILENAME: NO_ARTIFACT_YAML,
                },
            },
            id="test_no_artifact_models",
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("updated_catalog_config_map")
class TestArtifactCountsNoArtifacts:
    """Tests for models without any artifacts — artifactCounts should be omitted."""

    def test_model_without_artifacts_omits_artifact_counts(
        self: Self,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        Given a custom catalog source with models that have no model-artifact,
        When calling the detail endpoint for each model,
        Then artifactCounts should be absent from the response.
        """
        errors = []
        for model_name in NO_ARTIFACT_MODELS:
            detail = execute_get_command(
                url=f"{model_catalog_rest_url[0]}sources/{NO_ARTIFACT_CATALOG_ID}/models/{model_name}",
                headers=model_registry_rest_headers,
            )
            assert "name" in detail, f"Detail endpoint returned invalid data for {model_name}: {list(detail.keys())}"
            has_artifact_counts = "artifactCounts" in detail
            LOGGER.info(f"Model: {model_name}, has artifactCounts: {has_artifact_counts}")
            if has_artifact_counts:
                errors.append(f"{model_name}: expected artifactCounts key to be absent, got {detail['artifactCounts']}")

        assert not errors, "Models without artifacts should omit artifactCounts:\n" + "\n".join(
            f"  - {err}" for err in errors
        )
