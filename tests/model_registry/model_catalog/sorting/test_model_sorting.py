from typing import Self

import pytest
from kubernetes.dynamic import DynamicClient
from simple_logger.logger import get_logger

from tests.model_registry.model_catalog.constants import (
    REDHAT_AI_VALIDATED_UNESCAPED_CATALOG_NAME,
    VALIDATED_CATALOG_ID,
)
from tests.model_registry.model_catalog.sorting.utils import (
    get_model_latencies,
    validate_accuracy_sorting_against_database,
)
from tests.model_registry.model_catalog.utils import get_models_from_catalog_api

LOGGER = get_logger(name=__name__)

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
        RHOAIENG-36856: Test accuracy sorting for FindModels endpoint

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
        )

        assert validate_accuracy_sorting_against_database(
            admin_client=admin_client,
            api_response=response,
            sort_order=sort_order,
        )

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
        RHOAIENG-45852: Validate that recommendations parameter affects artifact-based model sorting

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
