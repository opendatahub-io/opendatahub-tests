from typing import Self

import pytest
import structlog
from kubernetes.dynamic import DynamicClient

from tests.ai_hub.model_catalog.constants import (
    RECOMMENDED_PARETO_ADDITIONAL_PARAMS,
    REDHAT_AI_VALIDATED_UNESCAPED_CATALOG_NAME,
    VALIDATED_CATALOG_ID,
)
from tests.ai_hub.model_catalog.sorting.utils import (
    RecommendedBaseline,
    assert_recommended_latency_ordering,
    get_all_recommended_model_names_paginated,
    get_model_latencies,
    get_recommended_and_legacy_model_names,
    get_recommended_model_names,
    validate_accuracy_sorting_against_database,
)
from tests.ai_hub.model_catalog.utils import get_models_from_catalog_api

LOGGER = structlog.get_logger(name=__name__)

pytestmark = [pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace")]


@pytest.mark.downstream_only
class TestAccuracySorting:
    """Test sorting for accuracy value in FindModels endpoint"""

    @pytest.mark.parametrize(
        "sort_order",
        [
            None,  # orderBy=ACCURACY without sortOrder
            "ASC",
            "DESC",
        ],
    )
    @pytest.mark.tier1
    def test_accuracy_sorting_works_correctly(
        self: Self,
        admin_client: DynamicClient,
        sort_order: str | None,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        Test accuracy sorting for FindModels endpoint

        This test validates accuracy sorting behavior with different sort_order parameters:

        When sort_order is None (orderBy=ACCURACY only):
        1. Models WITH accuracy appear first (in any order)
        2. Models WITHOUT accuracy appear after, sorted by ID in ASC order

        When sort_order is ASC or DESC (orderBy=ACCURACY&sortOrder=ASC/DESC):
        1. Models WITH accuracy appear first, sorted by accuracy value (ASC/DESC)
        2. Models WITHOUT accuracy appear after, sorted by ID in ASC order

        Validates both the presence of models and their correct ordering by comparing
        against direct database queries.
        """
        LOGGER.info(f"Testing accuracy sorting: sortOrder={sort_order}")

        response = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            order_by="ACCURACY",
            sort_order=sort_order,
            page_size=1000,
        )

        errors = validate_accuracy_sorting_against_database(
            admin_client=admin_client,
            api_response=response,
            sort_order=sort_order,
        )
        assert not errors, f"Accuracy sorting validation failed (sortOrder={sort_order}):\n" + "\n".join(errors)

    @pytest.mark.parametrize(
        "use_case",
        [
            "code_fixing",
            pytest.param("chatbot", marks=pytest.mark.tier1),  # Dashboard default use case
            "long_rag",
            "rag",
        ],
    )
    def test_recommendations_parameter_affects_artifact_sorting(
        self: Self,
        use_case: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        Validate that recommendations parameter affects artifact-based model sorting

        This test is parametrized by use_case and validates:
        1. Without recommendations=true: Models sorted by lowest latency across ALL artifacts
        2. With recommendations=true: Models sorted by lowest latency among ONLY recommended artifacts
        3. Both responses contain the same set of models
        4. Both responses are sorted in ascending order by their respective minimum latency values
        """
        LOGGER.info(f"Testing artifact sorting with and without recommendations parameter for use_case={use_case}")

        # Common filter and sort parameters
        artifact_property = "ttft_p90.double_value"
        artifact_filter = f"use_case.string_value='{use_case}'"

        # Get models sorted WITHOUT recommendations (all artifacts considered)
        response_all = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_label=REDHAT_AI_VALIDATED_UNESCAPED_CATALOG_NAME,
            order_by=f"artifacts.{artifact_property}",
            sort_order="ASC",
            additional_params=f"&filterQuery=artifacts.{artifact_filter}",
        )

        # Get models sorted WITH recommendations (only Pareto-optimal artifacts considered)
        response_recommended = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_label=REDHAT_AI_VALIDATED_UNESCAPED_CATALOG_NAME,
            order_by=f"artifacts.{artifact_property}",
            sort_order="ASC",
            additional_params=f"&filterQuery=artifacts.{artifact_filter}&recommendations=true",
        )

        # Extract model names preserving order
        all_model_names = [m["name"] for m in response_all["items"]]
        recommended_model_names = [m["name"] for m in response_recommended["items"]]

        LOGGER.info(f"Found {len(all_model_names)} models without recommendations filter")
        LOGGER.info(f"Found {len(recommended_model_names)} models with recommendations filter")

        # Validate that both queries return models
        assert all_model_names, "Should have models in response without recommendations"
        assert recommended_model_names, "Should have models in response with recommendations"

        assert set(all_model_names) == set(recommended_model_names), "Both responses should contain the same models"

        # Fetch actual minimum latency values for each model and validate ordering
        LOGGER.info("Fetching minimum latency values for models without recommendations filter")
        all_latencies = get_model_latencies(
            model_names=all_model_names,
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_id=VALIDATED_CATALOG_ID,
            property_field=artifact_property,
            artifact_filter_query=artifact_filter,
            sort_order="ASC",
            recommendations=False,
        )

        LOGGER.info("Fetching minimum latency values for models with recommendations filter")
        recommended_latencies = get_model_latencies(
            model_names=recommended_model_names,
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_id=VALIDATED_CATALOG_ID,
            property_field=artifact_property,
            artifact_filter_query=artifact_filter,
            sort_order="ASC",
            recommendations=True,
        )

        # Validate that latency values are in ascending order
        assert all_latencies == sorted(all_latencies), (
            f"Models without recommendations not sorted correctly by latency (ASC). "
            f"Expected order: {sorted(all_latencies)}, Actual order: {all_latencies}"
        )

        assert recommended_latencies == sorted(recommended_latencies), (
            f"Models with recommendations not sorted correctly by latency (ASC). "
            f"Expected order: {sorted(recommended_latencies)}, Actual order: {recommended_latencies}"
        )

        LOGGER.info("Validated that both responses are sorted correctly in ascending order")


@pytest.mark.downstream_only
class TestRecommendedSorting:
    """Test sorting for orderBy=RECOMMENDED in FindModels endpoint (RHOAIENG-68302).

    Acceptance criteria coverage:
    - AC1/3: test_recommended_sorting_matches_legacy_baseline (explicit ASC, default, no legacy param)
    - AC2: test_recommended_desc_sorting (legacy DESC equivalence, set match, latency ordering)
    - AC4: test_legacy_recommendations_backward_compatibility
    - AC5: test_recommended_sorting_matches_legacy_baseline (test_with_legacy_param)
    - AC6: test_recommended_sorting_with_pareto_parameters
    """

    def test_legacy_recommendations_backward_compatibility(
        self: Self,
        recommended_baseline: RecommendedBaseline,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        Given recommendations=true is used without orderBy=RECOMMENDED (RHOAIENG-68302 AC4)
        When the legacy baseline model order is inspected
        Then models are returned with recommended latencies in non-decreasing order
        """
        assert_recommended_latency_ordering(
            model_names=recommended_baseline.model_names,
            recommended_baseline=recommended_baseline,
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            expected_order="ASC",
        )

    @pytest.mark.parametrize(
        "sort_order, extra_params",
        [
            pytest.param("ASC", "", id="test_explicit_asc"),
            pytest.param(None, "", id="test_default_sort_order"),
            pytest.param("ASC", "&recommendations=true", id="test_with_legacy_param"),
        ],
    )
    def test_recommended_sorting_matches_legacy_baseline(
        self: Self,
        recommended_baseline: RecommendedBaseline,
        sort_order: str | None,
        extra_params: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        Given the legacy recommendations=true baseline
        When orderBy=RECOMMENDED is used with various sortOrder and parameter combinations
        Then the result matches the baseline and recommended latencies are non-decreasing
        """
        model_names = get_recommended_model_names(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            artifact_filter=recommended_baseline.artifact_filter,
            sort_order=sort_order,
            extra_params=extra_params,
        )

        assert model_names, "Should have models with orderBy=RECOMMENDED"

        assert model_names == recommended_baseline.model_names, (
            "orderBy=RECOMMENDED should match legacy recommendations=true baseline"
        )

        assert_recommended_latency_ordering(
            model_names=model_names,
            recommended_baseline=recommended_baseline,
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            expected_order="ASC",
        )

    def test_recommended_desc_sorting(
        self: Self,
        recommended_baseline: RecommendedBaseline,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        Given the legacy recommendations=true baseline (ASC order)
        When orderBy=RECOMMENDED with sortOrder=DESC is used
        Then legacy recommendations=true&sortOrder=DESC returns the same models in the same order,
        the same model set as the baseline is returned, latency values are non-increasing,
        and models without latency sort last
        """
        recommended_model_names, legacy_model_names = get_recommended_and_legacy_model_names(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            artifact_filter=recommended_baseline.artifact_filter,
            sort_order="DESC",
        )

        assert recommended_model_names, "Should have models with orderBy=RECOMMENDED sortOrder=DESC"
        assert legacy_model_names, "Should have models with recommendations=true sortOrder=DESC"

        assert legacy_model_names == recommended_model_names, (
            "recommendations=true&sortOrder=DESC should match orderBy=RECOMMENDED&sortOrder=DESC"
        )

        assert set(recommended_model_names) == set(recommended_baseline.model_names), (
            "ASC and DESC should return the same set of models"
        )

        assert_recommended_latency_ordering(
            model_names=recommended_model_names,
            recommended_baseline=recommended_baseline,
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            expected_order="DESC",
        )

    def test_recommended_pagination(
        self: Self,
        recommended_baseline: RecommendedBaseline,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        Given orderBy=RECOMMENDED returns multiple models
        When a small pageSize is used and the nextPageToken is followed
        Then all pages together contain the same models as the unpaginated baseline
        """
        small_page_size = 5
        max_pages = (len(recommended_baseline.model_names) + small_page_size - 1) // small_page_size

        all_model_names = get_all_recommended_model_names_paginated(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            artifact_filter=recommended_baseline.artifact_filter,
            page_size=small_page_size,
            max_pages=max_pages,
        )

        assert all_model_names == recommended_baseline.model_names, (
            "Paginated results should match the unpaginated baseline in order"
        )

    def test_recommended_sorting_with_pareto_parameters(
        self: Self,
        recommended_baseline: RecommendedBaseline,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        Given orderBy=RECOMMENDED with explicit Pareto parameters (targetRPS, latencyProperty, etc.)
        When compared to recommendations=true with the same parameters
        Then both return the same models in order and results stay within the default baseline set
        """
        recommended_model_names, legacy_model_names = get_recommended_and_legacy_model_names(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            artifact_filter=recommended_baseline.artifact_filter,
            sort_order="ASC",
            extra_params=RECOMMENDED_PARETO_ADDITIONAL_PARAMS,
        )

        assert recommended_model_names, "Should have models with orderBy=RECOMMENDED and Pareto parameters"
        assert legacy_model_names, "Should have models with recommendations=true and Pareto parameters"

        assert recommended_model_names == legacy_model_names, (
            "orderBy=RECOMMENDED with Pareto parameters should match legacy recommendations=true"
        )

        assert set(recommended_model_names) <= set(recommended_baseline.model_names), (
            "Pareto filtering should not return models outside the default recommendations baseline"
        )
