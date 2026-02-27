import pytest
from dictdiffer import diff
from simple_logger.logger import get_logger
from typing import Self, Any
from tests.model_registry.model_catalog.constants import (
    REDHAT_AI_CATALOG_ID,
    VALIDATED_CATALOG_ID,
    MODEL_ARTIFACT_TYPE,
    METRICS_ARTIFACT_TYPE,
    REDHAT_AI_CATALOG_NAME,
    REDHAT_AI_VALIDATED_UNESCAPED_CATALOG_NAME,
)
from tests.model_registry.model_catalog.utils import get_models_from_catalog_api
from tests.model_registry.model_catalog.search.utils import (
    fetch_all_artifacts_with_dynamic_paging,
    validate_model_contains_search_term,
    validate_search_results_against_database,
    validate_filter_query_results_against_database,
    validate_performance_data_files_on_pod,
    validate_model_artifacts_match_criteria_and,
    validate_model_artifacts_match_criteria_or,
)
from tests.model_registry.utils import get_model_catalog_pod
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError

LOGGER = get_logger(name=__name__)
pytestmark = [pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace")]


class TestSearchModelCatalog:
    @pytest.mark.smoke
    def test_search_model_catalog_source_label(
        self: Self,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        RHOAIENG-33656: Validate search model catalog by source label
        """

        redhat_ai_filter_moldels_size = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_label=REDHAT_AI_CATALOG_NAME,
        )["size"]
        redhat_ai_validated_filter_models_size = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_label=REDHAT_AI_VALIDATED_UNESCAPED_CATALOG_NAME,
        )["size"]
        no_filtered_models_size = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url, model_registry_rest_headers=model_registry_rest_headers
        )["size"]
        both_filtered_models_size = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_label=f"{REDHAT_AI_VALIDATED_UNESCAPED_CATALOG_NAME},{REDHAT_AI_CATALOG_NAME}",
        )["size"]
        LOGGER.info(f"no_filtered_models_size: {no_filtered_models_size}")
        assert no_filtered_models_size > 0
        assert no_filtered_models_size == both_filtered_models_size
        assert redhat_ai_filter_moldels_size + redhat_ai_validated_filter_models_size == both_filtered_models_size

    def test_search_model_catalog_invalid_source_label(
        self: Self,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        RHOAIENG-33656:
        Validate search model catalog by invalid source label
        """

        null_size = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_label="null",
        )["size"]

        invalid_size = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_label="invalid",
        )["size"]

        assert null_size == invalid_size == 0, (
            "Expected 0 models for null and invalid source label found {null_size} and {invalid_size}"
        )

    @pytest.mark.parametrize(
        "randomly_picked_model_from_catalog_api_by_source,source_filter",
        [
            pytest.param(
                {"source": VALIDATED_CATALOG_ID, "header_type": "registry"},
                REDHAT_AI_VALIDATED_UNESCAPED_CATALOG_NAME,
                id="test_search_model_catalog_redhat_ai_validated",
            ),
            pytest.param(
                {"source": REDHAT_AI_CATALOG_ID, "header_type": "registry"},
                REDHAT_AI_CATALOG_NAME,
                id="test_search_model_catalog_redhat_ai_default",
            ),
        ],
        indirect=["randomly_picked_model_from_catalog_api_by_source"],
    )
    def test_search_model_catalog_match(
        self: Self,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        randomly_picked_model_from_catalog_api_by_source: tuple[dict[Any, Any], str, str],
        source_filter: str,
    ):
        """
        RHOAIENG-33656: Validate search model catalog by match
        """
        random_model, random_model_name, _ = randomly_picked_model_from_catalog_api_by_source
        LOGGER.info(f"random_model_name: {random_model_name}")
        result = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_label=source_filter,
            additional_params=f"&filterQuery=name='{random_model_name}'",
        )
        assert random_model_name == result["items"][0]["name"]
        assert result["size"] == 1

        differences = list(diff(random_model, result["items"][0]))
        assert not differences, f"Expected no differences in model information for {random_model_name}: {differences}"
        LOGGER.info("Model information matches")


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
                {"header_type": "registry"},
                id="invalid_artifact_type",
            ),
        ],
        indirect=["randomly_picked_model_from_catalog_api_by_source"],
    )
    def test_error_handled_for_invalid_artifact_type(
        self: Self,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        randomly_picked_model_from_catalog_api_by_source: tuple[dict[Any, Any], str, str],
    ):
        """
        RHOAIENG-33659: Validates that the API returns the correct error when an invalid artifactType
        is provided regardless of catalog or model.
        """
        invalid_artifact_type = "invalid"
        _, model_name, catalog_id = randomly_picked_model_from_catalog_api_by_source
        invalid_error = f"artifactType: invalid value '{invalid_artifact_type}' for ArtifactTypeQueryParam"

        LOGGER.info(f"Testing invalid artifact type: '{invalid_artifact_type}'")

        with pytest.raises(ResourceNotFoundError, match=invalid_error):
            fetch_all_artifacts_with_dynamic_paging(
                url_with_pagesize=(
                    f"{model_catalog_rest_url[0]}sources/{catalog_id}/models/{model_name}/artifacts?"
                    f"artifactType={invalid_artifact_type}&pageSize"
                ),
                headers=model_registry_rest_headers,
                page_size=1,
            )

        LOGGER.info(f"Successfully validated that invalid artifact type '{invalid_artifact_type}' raises an error")

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


class TestSearchModelCatalogQParameter:
    """Test suite for the 'q' search parameter functionality (RHOAIENG-36911)."""

    @pytest.mark.parametrize(
        "search_term",
        [
            "deepseek",
            "red hat",
            "granite-8b",
            pytest.param(
                "The Llama 4 collection of models are natively multimodal AI models that enable text and multimodal experiences. These models leverage a mixture-of-experts architecture to offer industry-leading performance in text and image understanding. These Llama 4 models mark the beginning of a new era for the Llama ecosystem. We are launching two efficient models in the Llama 4 series, Llama 4 Scout, a 17 billion parameter model with 16 experts, and Llama 4 Maverick, a 17 billion parameter model with 128 experts.",  # noqa: E501
                id="long_description",
            ),
        ],
    )
    def test_q_parameter_basic_search(
        self: Self,
        search_term: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        model_registry_namespace: str,
    ):
        """Test basic search functionality with q parameter using database validation"""
        LOGGER.info(f"Testing search for term: {search_term}")

        response = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            q=search_term,
        )

        assert "items" in response
        models = response.get("items", [])

        LOGGER.info(f"Found {len(models)} models for search term '{search_term}'")

        # Validate API results against database query
        is_valid, errors = validate_search_results_against_database(
            api_response=response,
            search_term=search_term,
            namespace=model_registry_namespace,
        )

        assert is_valid, f"API search results do not match database query for '{search_term}': {errors}"

        # Additional validation: ensure returned models actually contain the search term
        for model in models:
            assert validate_model_contains_search_term(model, search_term), (
                f"Model '{model.get('name')}' doesn't contain search term '{search_term}' in any searchable field"
            )

    @pytest.mark.parametrize(
        "search_term,case_variant", [("granite", "GRANITE"), ("text", "TEXT"), ("deepseek", "DeepSeek")]
    )
    def test_q_parameter_case_insensitive(
        self: Self,
        search_term: str,
        case_variant: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        model_registry_namespace: str,
    ):
        """Test that search is case insensitive using database validation"""
        LOGGER.info(f"Testing case insensitivity: '{search_term}' vs '{case_variant}'")

        response1 = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            q=search_term,
        )

        response2 = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            q=case_variant,
        )

        # Validate both responses against database
        is_valid1, errors1 = validate_search_results_against_database(
            api_response=response1,
            search_term=search_term,
            namespace=model_registry_namespace,
        )
        assert is_valid1, f"API search results do not match database query for '{search_term}': {errors1}"

        is_valid2, errors2 = validate_search_results_against_database(
            api_response=response2,
            search_term=case_variant,
            namespace=model_registry_namespace,
        )
        assert is_valid2, f"API search results do not match database query for '{case_variant}': {errors2}"

        models1 = response1.get("items", [])
        models2 = response2.get("items", [])

        model_ids1 = sorted([m.get("id") for m in models1])
        model_ids2 = sorted([m.get("id") for m in models2])

        assert model_ids1 == model_ids2, (
            f"Case insensitive search failed:\n"
            f"'{search_term}' returned {len(models1)} models\n"
            f"'{case_variant}' returned {len(models2)} models"
        )

    def test_q_parameter_no_results(
        self: Self,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        model_registry_namespace: str,
    ):
        """Test search with term that should return no results using database validation"""
        nonexistent_term = "nonexistent_search_term_12345_abcdef"
        LOGGER.info(f"Testing search for nonexistent term: {nonexistent_term}")

        response = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            q=nonexistent_term,
        )

        # Validate API results against database query
        is_valid, errors = validate_search_results_against_database(
            api_response=response,
            search_term=nonexistent_term,
            namespace=model_registry_namespace,
        )
        assert is_valid, f"API search results do not match database query for '{nonexistent_term}': {errors}"

        models = response.get("items", [])
        assert len(models) == 0, f"Expected no results for '{nonexistent_term}', got {len(models)} models"

    @pytest.mark.parametrize("search_term", ["", None])
    def test_q_parameter_empty_query(
        self: Self,
        search_term,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Test behavior with empty or None q parameter using database validation"""
        LOGGER.info(f"Testing empty query: {repr(search_term)}")

        response = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            q=search_term,
        )

        models = response.get("items", [])
        LOGGER.info(f"Empty/None query returned {len(models)} models")

    def test_q_parameter_with_source_label_filter(
        self: Self,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Test q parameter combined with source_label filtering using database validation"""
        search_term = "granite"
        source_label = REDHAT_AI_CATALOG_NAME

        LOGGER.info(f"Testing combined search: q='{search_term}' with sourceLabel='{source_label}'")

        response = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            q=search_term,
            source_label=source_label,
        )

        models = response.get("items", [])
        LOGGER.info(f"Combined filter returned {len(models)} models")

        # Validate that all returned models match the search term (the search part of the combined query)
        for model in models:
            assert validate_model_contains_search_term(model, search_term), (
                f"Model '{model.get('name')}' doesn't contain search term '{search_term}'"
            )

        # Get search results without source filter to compare subset relationship
        search_only_response = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            q=search_term,
        )

        # Combined filter results should be a subset of search-only results
        search_only_model_ids = set(m.get("id") for m in search_only_response.get("items", []))
        combined_model_ids = set(m.get("id") for m in models)

        assert combined_model_ids.issubset(search_only_model_ids), (
            f"Combined filter results should be a subset of search-only results. "
            f"Extra models in combined: {combined_model_ids - search_only_model_ids}"
        )


class TestSearchModelsByFilterQuery:
    def test_search_models_by_filter_query(
        self: Self,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        model_registry_namespace: str,
    ):
        """
        RHOAIENG-33658: Tests that the API returns all models matching a given filter query and
        that the database results are consistent.
        """
        # Filter parameters
        licenses = "'gemma','modified-mit'"
        language_pattern_1 = "%iT%"
        language_pattern_2 = "%de%"

        # using ILIKE for case-insensitive matching
        filter_query = f"license IN ({licenses}) AND (language ILIKE '{language_pattern_1}' \
            OR language ILIKE '{language_pattern_2}')"

        result = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            additional_params=f"&filterQuery={filter_query}",
        )

        # Validate API results against database query using same parameters
        is_valid, errors = validate_filter_query_results_against_database(
            api_response=result,
            licenses=licenses,
            language_pattern_1=language_pattern_1,
            language_pattern_2=language_pattern_2,
            namespace=model_registry_namespace,
        )

        assert is_valid, f"API filter query results do not match database query: {errors}"

        # Additional validation: ensure returned models match the filter criteria
        for item in result["items"]:
            assert item["license"] in licenses, f"Item license {item['license']} not in {licenses}"
            assert any(language in item["language"] for language in ["it", "de"]), (
                f"Item language {item['language']} not in ['it', 'de']"
            )

        LOGGER.info("All models match the filter query and database validation passed")

    def test_search_models_by_invalid_filter_query(
        self: Self,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        model_registry_namespace: str,
    ):
        """
        RHOAIENG-36938: Tests the API's response to invalid and non-matching filter queries.
        It verifies that an invalid filter query raises the correct error and
        that a query with no matches returns zero models.
        """
        non_existing_filter_query = "fake IN ('gemma','modified-mit'))"
        with pytest.raises(ResourceNotFoundError, match="invalid filter query"):
            get_models_from_catalog_api(
                model_catalog_rest_url=model_catalog_rest_url,
                model_registry_rest_headers=model_registry_rest_headers,
                additional_params=f"&filterQuery={non_existing_filter_query}",
            )
        # Test with a valid filter query that should return zero results
        no_result_licenses = "'fake'"
        no_result_filter_query = f"license IN ({no_result_licenses})"
        result = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            additional_params=f"&filterQuery={no_result_filter_query}",
        )
        LOGGER.info(f"Result: {result['size']}")
        assert result["size"] == 0, "Expected 0 models for a non-existing filter query"

        # Validate API results against database query using same license parameter
        is_valid, errors = validate_filter_query_results_against_database(
            api_response=result,
            licenses=no_result_licenses,
            namespace=model_registry_namespace,
        )
        assert is_valid, f"API filter query results do not match database query: {errors}"

    # Performance data are available only in downstream
    @pytest.mark.downstream_only
    def test_presence_performance_data_on_pod(
        self: Self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
    ):
        """
        RHOAIENG-36938: Checks that performance data files exist for all models in the catalog pod.
        It ensures that each model has the required metadata and performance files present in the pod.
        """

        model_catalog_pod = get_model_catalog_pod(
            client=admin_client, model_registry_namespace=model_registry_namespace
        )[0]

        validation_results = validate_performance_data_files_on_pod(model_catalog_pod=model_catalog_pod)

        # Assert that all models have all required performance data files
        assert not validation_results, f"Models with missing performance data files: {validation_results}"

    @pytest.mark.parametrize(
        "models_from_filter_query, expected_value, logic_type",
        [
            pytest.param(
                "artifacts.requests_per_second > 15.0",
                [{"key_name": "requests_per_second", "key_type": "double_value", "comparison": "min", "value": 15.0}],
                "and",
                id="performance_min_filter",
            ),
            pytest.param(
                "artifacts.hardware_count = 8",
                [{"key_name": "hardware_count", "key_type": "int_value", "comparison": "exact", "value": 8}],
                "and",
                id="hardware_exact_filter",
            ),
            pytest.param(
                "(artifacts.hardware_type LIKE 'H200') AND (artifacts.ttft_p95 < 50)",
                [
                    {"key_name": "hardware_type", "key_type": "string_value", "comparison": "exact", "value": "H200"},
                    {"key_name": "ttft_p95", "key_type": "double_value", "comparison": "max", "value": 50},
                ],
                "and",
                id="test_combined_hardware_performance_filter_mixed_types",
            ),
            pytest.param(
                "(artifacts.ttft_mean < 100) AND (artifacts.requests_per_second > 10)",
                [
                    {"key_name": "ttft_mean", "key_type": "double_value", "comparison": "max", "value": 100},
                    {"key_name": "requests_per_second", "key_type": "double_value", "comparison": "min", "value": 10},
                ],
                "and",
                id="test_combined_hardware_performance_filter_numeric_types",
            ),
            pytest.param(
                "(artifacts.tps_mean < 247) OR (artifacts.hardware_type LIKE 'A100-80')",
                [
                    {"key_name": "tps_mean", "key_type": "double_value", "comparison": "max", "value": 247},
                    {
                        "key_name": "hardware_type",
                        "key_type": "string_value",
                        "comparison": "exact",
                        "value": "A100-80",
                    },
                ],
                "or",
                id="performance_or_hardware_filter",
            ),
        ],
        indirect=["models_from_filter_query"],
    )
    def test_filter_query_advanced_model_search(
        self: Self,
        models_from_filter_query: list[str],
        expected_value: list[dict[str, Any]],
        logic_type: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        RHOAIENG-39615: Advanced filter query test for performance-based filtering with AND/OR logic
        """
        errors = []

        # Additional validation: ensure returned models match the filter criteria
        for model_name in models_from_filter_query:
            url = f"{model_catalog_rest_url[0]}sources/{VALIDATED_CATALOG_ID}/models/{model_name}/artifacts?pageSize"
            LOGGER.info(f"Validating model: {model_name} with {len(expected_value)} {logic_type.upper()} validation(s)")

            # Fetch all artifacts with dynamic page size adjustment
            all_model_artifacts = fetch_all_artifacts_with_dynamic_paging(
                url_with_pagesize=url,
                headers=model_registry_rest_headers,
                page_size=200,
            )["items"]

            validation_result = None
            # Select validation function based on logic type
            if logic_type == "and":
                validation_result = validate_model_artifacts_match_criteria_and(
                    all_model_artifacts=all_model_artifacts, expected_validations=expected_value, model_name=model_name
                )
            elif logic_type == "or":
                validation_result = validate_model_artifacts_match_criteria_or(
                    all_model_artifacts=all_model_artifacts, expected_validations=expected_value, model_name=model_name
                )
            else:
                raise ValueError(f"Invalid logic_type: {logic_type}. Must be 'and' or 'or'")

            if validation_result:
                LOGGER.info(f"For Model: {model_name}, {logic_type.upper()} validation completed successfully")
            else:
                errors.append(model_name)

        assert not errors, f"{logic_type.upper()} filter validations failed for {', '.join(errors)}"
        LOGGER.info(
            f"Advanced {logic_type.upper()} filter validation completed for {len(models_from_filter_query)} models"
        )
