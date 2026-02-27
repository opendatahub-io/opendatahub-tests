import pytest
from typing import Self, Any
import random
from tests.model_registry.model_catalog.search.utils import (
    fetch_all_artifacts_with_dynamic_paging,
    validate_model_artifacts_match_criteria_and,
    validate_model_artifacts_match_criteria_or,
    validate_recommendations_subset,
)
from tests.model_registry.model_catalog.constants import (
    VALIDATED_CATALOG_ID,
)
from kubernetes.dynamic.exceptions import ResourceNotFoundError
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
        "randomly_picked_model_from_catalog_api_by_source, invalid_filter_query",
        [
            pytest.param(
                {"catalog_id": VALIDATED_CATALOG_ID, "header_type": "registry"},
                "fake IN ('test', 'fake'))",
                id="test_invalid_artifact_filter_query_malformed",
            ),
            pytest.param(
                {"catalog_id": VALIDATED_CATALOG_ID, "header_type": "registry"},
                "ttft_p90.double_value < abc",
                id="test_invalid_artifact_filter_query_data_type_mismatch",
            ),
            pytest.param(
                {"catalog_id": VALIDATED_CATALOG_ID, "header_type": "registry"},
                "hardware_type.string_value = 5.0",
                id="test_invalid_artifact_filter_query_data_type_mismatch_equality",
            ),
        ],
        indirect=["randomly_picked_model_from_catalog_api_by_source"],
    )
    def test_search_artifacts_by_invalid_filter_query(
        self: Self,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        randomly_picked_model_from_catalog_api_by_source: tuple[dict, str, str],
        invalid_filter_query: str,
    ):
        """
        Tests the API's response to invalid filter queries syntax when searching artifacts.
        It verifies that an invalid filter query syntax raises the correct error.
        """
        _, model_name, catalog_id = randomly_picked_model_from_catalog_api_by_source

        LOGGER.info(f"Testing invalid artifact filter query: '{invalid_filter_query}' for model: {model_name}")
        with pytest.raises(ResourceNotFoundError, match="invalid filter query"):
            fetch_all_artifacts_with_dynamic_paging(
                url_with_pagesize=(
                    f"{model_catalog_rest_url[0]}sources/{catalog_id}/models/{model_name}/artifacts?"
                    f"filterQuery={invalid_filter_query}&pageSize"
                ),
                headers=model_registry_rest_headers,
                page_size=1,
            )

        LOGGER.info(
            f"Successfully validated that invalid artifact filter query '{invalid_filter_query}' raises an error"
        )

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
