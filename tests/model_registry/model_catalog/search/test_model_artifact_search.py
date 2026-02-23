import pytest
from typing import Self, Any
import random
from dictdiffer import diff

from tests.model_registry.model_catalog.search.utils import (
    fetch_all_artifacts_with_dynamic_paging,
    validate_model_artifacts_match_criteria_and,
    validate_model_artifacts_match_criteria_or,
    validate_recommendations_subset,
)
from tests.model_registry.model_catalog.constants import (
    VALIDATED_CATALOG_ID,
    MODEL_ARTIFACT_TYPE,
    METRICS_ARTIFACT_TYPE,
)
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)
pytestmark = [pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace")]
MODEL_NAMEs_ARTIFACT_SEARCH: list[str] = [
    "RedHatAI/Llama-3.1-8B-Instruct",
    "RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic",
    "RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16",
    "RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w8a8",
    "RedHatAI/Mixtral-8x7B-Instruct-v0.1",
]


class TestSearchArtifactsByFilterQuery:
    @pytest.mark.parametrize(
        "randomly_picked_model_from_catalog_api_by_source, filter_query, expected_value, logic_type",
        [
            pytest.param(
                {
                    "catalog_id": VALIDATED_CATALOG_ID,
                    "header_type": "registry",
                    "model_name": random.choice(MODEL_NAMEs_ARTIFACT_SEARCH),
                },
                "hardware_type.string_value = 'ABC-1234'",
                None,
                None,
                id="test_valid_artifact_filter_query_no_results",
            ),
            pytest.param(
                {
                    "catalog_id": VALIDATED_CATALOG_ID,
                    "header_type": "registry",
                    "model_name": random.choice(MODEL_NAMEs_ARTIFACT_SEARCH),
                },
                "requests_per_second.double_value > 15.0",
                [{"key_name": "requests_per_second", "key_type": "double_value", "comparison": "min", "value": 15.0}],
                "and",
                id="test_performance_min_filter",
            ),
            pytest.param(
                {
                    "catalog_id": VALIDATED_CATALOG_ID,
                    "header_type": "registry",
                    "model_name": random.choice(MODEL_NAMEs_ARTIFACT_SEARCH),
                },
                "hardware_count.int_value = 8",
                [{"key_name": "hardware_count", "key_type": "int_value", "comparison": "exact", "value": 8}],
                "and",
                id="test_hardware_exact_filter",
            ),
            pytest.param(
                {
                    "catalog_id": VALIDATED_CATALOG_ID,
                    "header_type": "registry",
                    "model_name": random.choice(MODEL_NAMEs_ARTIFACT_SEARCH),
                },
                "(hardware_type.string_value = 'H100') AND (ttft_p99.double_value < 200)",
                [
                    {"key_name": "hardware_type", "key_type": "string_value", "comparison": "exact", "value": "H100"},
                    {"key_name": "ttft_p99", "key_type": "double_value", "comparison": "max", "value": 199},
                ],
                "and",
                id="test_combined_hardware_performance_filter_and_operation",
            ),
            pytest.param(
                {
                    "catalog_id": VALIDATED_CATALOG_ID,
                    "header_type": "registry",
                    "model_name": random.choice(MODEL_NAMEs_ARTIFACT_SEARCH),
                },
                "(tps_mean.double_value <260) OR (hardware_type.string_value = 'A100-80')",
                [
                    {"key_name": "tps_mean", "key_type": "double_value", "comparison": "max", "value": 260},
                    {
                        "key_name": "hardware_type",
                        "key_type": "string_value",
                        "comparison": "exact",
                        "value": "A100-80",
                    },
                ],
                "or",
                id="performance_or_hardware_filter_or_operation",
            ),
        ],
        indirect=["randomly_picked_model_from_catalog_api_by_source"],
    )
    def test_filter_query_advanced_artifact_search(
        self: Self,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        randomly_picked_model_from_catalog_api_by_source: tuple[dict, str, str],
        filter_query: str,
        expected_value: list[dict[str, Any]] | None,
        logic_type: str | None,
    ):
        """
        Advanced filter query test for artifact-based filtering with AND/OR logic
        """
        _, model_name, catalog_id = randomly_picked_model_from_catalog_api_by_source

        LOGGER.info(f"Testing artifact filter query: '{filter_query}' for model: {model_name}")

        result = fetch_all_artifacts_with_dynamic_paging(
            url_with_pagesize=(
                f"{model_catalog_rest_url[0]}sources/{catalog_id}/models/{model_name}/artifacts?"
                f"filterQuery={filter_query}&pageSize"
            ),
            headers=model_registry_rest_headers,
            page_size=100,
        )

        if expected_value is None:
            # Simple validation of length and size for basic filter queries
            assert result["items"] == [], f"Filter query '{filter_query}' should return valid results"
            assert result["size"] == 0, f"Size should be 0 for filter query '{filter_query}'"
            LOGGER.info(
                f"Successfully validated that filter query '{filter_query}' returns {len(result['items'])} artifacts"
            )
        else:
            # Advanced validation using criteria matching
            all_artifacts = result["items"]

            validation_result = None
            # Select validation function based on logic type
            if logic_type == "and":
                validation_result = validate_model_artifacts_match_criteria_and(
                    all_model_artifacts=all_artifacts, expected_validations=expected_value, model_name=model_name
                )
            elif logic_type == "or":
                validation_result = validate_model_artifacts_match_criteria_or(
                    all_model_artifacts=all_artifacts, expected_validations=expected_value, model_name=model_name
                )
            else:
                raise ValueError(f"Invalid logic_type: {logic_type}. Must be 'and' or 'or'")

            if validation_result:
                LOGGER.info(
                    f"For Model: {model_name}, {logic_type} validation completed successfully"
                    f" for {len(all_artifacts)} artifacts"
                )
            else:
                pytest.fail(f"{logic_type} filter validation failed for model {model_name}")

    @pytest.mark.parametrize(
        "randomly_picked_model_from_catalog_api_by_source",
        [
            pytest.param(
                {
                    "catalog_id": VALIDATED_CATALOG_ID,
                    "header_type": "registry",
                    "model_name": random.choice(MODEL_NAMEs_ARTIFACT_SEARCH),
                },
                id="test_performance_artifacts_recommendations_parameter",
            ),
        ],
        indirect=["randomly_picked_model_from_catalog_api_by_source"],
    )
    def test_performance_artifacts_recommendations_parameter(
        self: Self,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        randomly_picked_model_from_catalog_api_by_source: tuple[dict, str, str],
    ):
        """
        Test the recommendations query parameter for performance artifacts endpoint.

        Validates that recommendations=true returns a filtered subset of performance
        artifacts that are optimal based on cost and latency compared to the full set.
        """
        _, model_name, catalog_id = randomly_picked_model_from_catalog_api_by_source

        LOGGER.info(f"Testing performance artifacts recommendations parameter for model: {model_name}")

        # Get all performance artifacts (baseline)
        full_results = fetch_all_artifacts_with_dynamic_paging(
            url_with_pagesize=(
                f"{model_catalog_rest_url[0]}sources/{catalog_id}/models/{model_name}/artifacts/performance?pageSize"
            ),
            headers=model_registry_rest_headers,
            page_size=100,
        )

        # Get recommendations-filtered performance artifacts
        recommendations_results = fetch_all_artifacts_with_dynamic_paging(
            url_with_pagesize=(
                f"{model_catalog_rest_url[0]}sources/{catalog_id}/models/{model_name}/artifacts/performance?"
                f"recommendations=true&pageSize"
            ),
            headers=model_registry_rest_headers,
            page_size=100,
        )

        if (full_results and not recommendations_results) or (len(recommendations_results) > len(full_results)):
            pytest.fail(f"Recommendations parameter functionality failed for model {model_name}")

        # Validate subset relationship
        validation_passed = validate_recommendations_subset(
            full_artifacts=full_results["items"],
            recommendations_artifacts=recommendations_results["items"],
            model_name=model_name,
        )

        assert validation_passed, f"Recommendations subset validation failed for model {model_name}"
        LOGGER.info(f"Successfully validated recommendations parameter functionality for model {model_name}")


class TestSearchModelArtifact:
    @pytest.mark.parametrize(
        "randomly_picked_model_from_catalog_api_by_source, artifact_type",
        [
            pytest.param(
                {"catalog_id": VALIDATED_CATALOG_ID, "header_type": "registry"},
                MODEL_ARTIFACT_TYPE,
                id="validated_model_artifact",
            ),
            pytest.param(
                {"catalog_id": VALIDATED_CATALOG_ID, "header_type": "registry"},
                METRICS_ARTIFACT_TYPE,
                id="validated_metrics_artifact",
            ),
        ],
        indirect=["randomly_picked_model_from_catalog_api_by_source"],
    )
    def test_validate_model_artifacts_by_artifact_type(
        self: Self,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        randomly_picked_model_from_catalog_api_by_source: tuple[dict[Any, Any], str, str],
        artifact_type: str,
    ):
        """
        RHOAIENG-33659: Validates that the model artifacts returned by the artifactType filter
        match the complete set of artifacts for a random model.
        """
        _, model_name, catalog_id = randomly_picked_model_from_catalog_api_by_source
        LOGGER.info(f"Artifact type: '{artifact_type}'")

        # Fetch all artifacts with dynamic page size adjustment
        all_model_artifacts = fetch_all_artifacts_with_dynamic_paging(
            url_with_pagesize=f"{model_catalog_rest_url[0]}sources/{catalog_id}/models/{model_name}/artifacts?pageSize",
            headers=model_registry_rest_headers,
            page_size=100,
        )["items"]

        # Fetch filtered artifacts by type with dynamic page size adjustment
        artifact_type_artifacts = fetch_all_artifacts_with_dynamic_paging(
            url_with_pagesize=(
                f"{model_catalog_rest_url[0]}sources/{catalog_id}/models/{model_name}/artifacts?"
                f"artifactType={artifact_type}&pageSize"
            ),
            headers=model_registry_rest_headers,
            page_size=50,
        )["items"]

        # Create lookup for validation
        all_artifacts_by_id = {artifact["id"]: artifact for artifact in all_model_artifacts}

        # Verify all filtered artifacts exist
        for artifact in artifact_type_artifacts:
            artifact_id = artifact["id"]
            assert artifact_id in all_artifacts_by_id, (
                f"Filtered artifact {artifact_id} not found in complete artifact list for {model_name}"
            )

            differences = list(diff(artifact, all_artifacts_by_id[artifact_id]))
            assert not differences, f"Artifact {artifact_id} mismatch for {model_name}: {differences}"

        # Verify the filter didn't miss any artifacts of the type
        artifacts_of_type_in_all = [
            artifact for artifact in all_model_artifacts if artifact.get("artifactType") == artifact_type
        ]
        assert len(artifact_type_artifacts) == len(artifacts_of_type_in_all), (
            f"Filter returned {len(artifact_type_artifacts)} {artifact_type} artifacts, "
            f"but found {len(artifacts_of_type_in_all)} in complete list for {model_name}"
        )

        LOGGER.info(f"Validated {len(artifact_type_artifacts)} {artifact_type} artifacts for {model_name}")

    @pytest.mark.parametrize(
        "randomly_picked_model_from_catalog_api_by_source",
        [
            pytest.param(
                {"catalog_id": VALIDATED_CATALOG_ID, "header_type": "registry"},
                id="validated_catalog",
            ),
        ],
        indirect=["randomly_picked_model_from_catalog_api_by_source"],
    )
    def test_multiple_artifact_type_filtering(
        self: Self,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        randomly_picked_model_from_catalog_api_by_source: tuple[dict[Any, Any], str, str],
    ):
        """
        RHOAIENG-33659: Validates that the API returns all artifacts of a random model
        when filtering by multiple artifact types.
        """
        _, model_name, catalog_id = randomly_picked_model_from_catalog_api_by_source
        artifact_types = f"artifactType={METRICS_ARTIFACT_TYPE},{MODEL_ARTIFACT_TYPE}"
        LOGGER.info(f"Testing multiple artifact types: '{artifact_types}'")
        # Fetch all artifacts with dynamic page size adjustment
        all_model_artifacts = fetch_all_artifacts_with_dynamic_paging(
            url_with_pagesize=f"{model_catalog_rest_url[0]}sources/{catalog_id}/models/{model_name}/artifacts?pageSize",
            headers=model_registry_rest_headers,
            page_size=100,
        )["items"]

        # Fetch filtered artifacts by type with dynamic page size adjustment
        artifact_type_artifacts = fetch_all_artifacts_with_dynamic_paging(
            url_with_pagesize=(
                f"{model_catalog_rest_url[0]}sources/{catalog_id}/models/{model_name}/artifacts?"
                f"{artifact_types}&pageSize"
            ),
            headers=model_registry_rest_headers,
            page_size=100,
        )["items"]

        assert len(artifact_type_artifacts) == len(all_model_artifacts), (
            f"Filter returned {len(artifact_type_artifacts)} artifacts, "
            f"but found {len(all_model_artifacts)} in complete list for {model_name}"
        )
