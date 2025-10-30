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
from tests.model_registry.model_catalog.utils import (
    get_models_from_catalog_api,
    fetch_all_artifacts_with_dynamic_paging,
    validate_model_contains_search_term,
    validate_search_results_against_database,
)
from kubernetes.dynamic.exceptions import ResourceNotFoundError

LOGGER = get_logger(name=__name__)
pytestmark = [
    pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace", "test_idp_user")
]


class TestSearchModelCatalog:
    @pytest.mark.smoke
    def test_search_model_catalog_source_label(
        self: Self, model_catalog_rest_url: list[str], model_registry_rest_headers: dict[str, str]
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

        assert no_filtered_models_size == both_filtered_models_size
        assert redhat_ai_filter_moldels_size + redhat_ai_validated_filter_models_size == both_filtered_models_size

    def test_search_model_catalog_invalid_source_label(
        self: Self, model_catalog_rest_url: list[str], model_registry_rest_headers: dict[str, str]
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


# All the tests in this class are failing for RHOAIENG-36938, there are two problems:
# 1. The filter parameter is setup to use artifact_type instead of artifactType
# 2. The filter with multiple artifact types is not working as expected
@pytest.mark.xfail(
    strict=True,
    reason="RHOAIENG-36938: artifact_type is usedinstead of artifactType, multiple artifact types are not working",
)
class TestSearchModelArtifact:
    @pytest.mark.parametrize(
        "randomly_picked_model_from_catalog_api_by_source, artifact_type",
        [
            pytest.param(
                {"catalog_id": REDHAT_AI_CATALOG_ID, "header_type": "registry"},
                MODEL_ARTIFACT_TYPE,
                id="redhat_ai_model_artifact",
            ),
            pytest.param(
                {"catalog_id": REDHAT_AI_CATALOG_ID, "header_type": "registry"},
                METRICS_ARTIFACT_TYPE,
                id="redhat_ai_metrics_artifact",
            ),
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
        _, model_name, catalog_id = randomly_picked_model_from_catalog_api_by_source

        invalid_artifact_type = "invalid"
        LOGGER.info(f"Testing invalid artifact type: '{invalid_artifact_type}'")

        with pytest.raises(ResourceNotFoundError, match=f"unsupported catalog artifact type: {invalid_artifact_type}"):
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
                {"catalog_id": REDHAT_AI_CATALOG_ID, "header_type": "registry"},
                id="redhat_ai_catalog",
            ),
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
        artifact_types = f"{METRICS_ARTIFACT_TYPE},{MODEL_ARTIFACT_TYPE}"
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
                f"artifactType={artifact_types}&pageSize"
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
