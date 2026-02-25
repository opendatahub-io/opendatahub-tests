from typing import Self

import pytest
from kubernetes.dynamic import DynamicClient
from simple_logger.logger import get_logger

from tests.model_registry.model_catalog.db_constants import (
    API_COMPUTED_FILTER_FIELDS,
    API_EXCLUDED_FILTER_FIELDS,
    FILTER_OPTIONS_DB_QUERY,
)
from tests.model_registry.model_catalog.metadata.utils import (
    compare_filter_options_with_database,
)
from tests.model_registry.model_catalog.utils import (
    execute_database_query,
    parse_psql_output,
)
from tests.model_registry.utils import execute_get_command, get_rest_headers
from utilities.user_utils import UserTestSession

LOGGER = get_logger(name=__name__)

pytestmark = [
    pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace", "original_user")
]


class TestFilterOptionsEndpoint:
    """
    Test class for validating the models/filter_options endpoint
    RHOAIENG-36696
    """

    # Cannot use non-admin user for this test as it cannot list the pods in the namespace
    @pytest.mark.parametrize(
        "user_token_for_api_calls,",
        [
            pytest.param(
                {},
                id="test_filter_options_admin_user",
            ),
            pytest.param(
                {"user_type": "sa_user"},
                id="test_filter_options_service_account",
            ),
        ],
        indirect=["user_token_for_api_calls"],
    )
    def test_comprehensive_coverage_against_database(
        self: Self,
        admin_client: DynamicClient,
        model_catalog_rest_url: list[str],
        user_token_for_api_calls: str,
        model_registry_namespace: str,
    ):
        """
        Validate filter options are comprehensive across all sources/models in DB.
        Acceptance Criteria: The returned options are comprehensive and not limited to a
        subset of models or a single source.

        This test executes the exact same SQL query the API uses and compares results
        to catch any discrepancies between database content and API response.

        Expected failure because of RHOAIENG-37069 & RHOAIENG-37226
        """
        api_url = f"{model_catalog_rest_url[0]}models/filter_options"
        LOGGER.info(f"Testing comprehensive database coverage for: {api_url}")

        api_response = execute_get_command(
            url=api_url,
            headers=get_rest_headers(token=user_token_for_api_calls),
        )

        api_filters = api_response["filters"]
        LOGGER.info(f"API returned {len(api_filters)} filter properties: {list(api_filters.keys())}")

        LOGGER.info(f"Executing database query in namespace: {model_registry_namespace}")

        db_result = execute_database_query(
            admin_client=admin_client, query=FILTER_OPTIONS_DB_QUERY, namespace=model_registry_namespace
        )
        parsed_result = parse_psql_output(psql_output=db_result)

        db_properties = parsed_result.get("properties", {})
        LOGGER.info(f"Raw database query returned {len(db_properties)} properties: {list(db_properties.keys())}")

        # Remove API-computed fields from API response before comparison
        filtered_api_filters = {
            key: value for key, value in api_filters.items() if key not in API_COMPUTED_FILTER_FIELDS
        }

        is_valid, comparison_errors = compare_filter_options_with_database(
            api_filters=filtered_api_filters,
            db_properties=db_properties,
            excluded_fields=API_EXCLUDED_FILTER_FIELDS,
        )

        if not is_valid:
            failure_msg = "Filter options API response does not match database content"
            failure_msg += "\nDetailed comparison errors:\n" + "\n".join(comparison_errors)
            assert False, failure_msg

        LOGGER.info("Comprehensive database coverage validation passed - API matches database exactly")

    @pytest.mark.parametrize(
        "user_token_for_api_calls,",
        [
            pytest.param(
                {},
                id="test_named_queries_admin_user",
            ),
            pytest.param(
                {"user_type": "test"},
                id="test_named_queries_non_admin_user",
            ),
            pytest.param(
                {"user_type": "sa_user"},
                id="test_named_queries_service_account",
            ),
        ],
        indirect=["user_token_for_api_calls"],
    )
    def test_named_queries_in_filter_options(
        self: Self,
        model_catalog_rest_url: list[str],
        user_token_for_api_calls: str,
        test_idp_user: UserTestSession,
    ):
        """
        Test for: RHOAIENG-38836
        Validate that namedQueries field is present in filter_options response.
        Validates that default-performance-filters named query exists with expected properties.
        """
        default_performance_filters = "default-performance-filters"
        url = f"{model_catalog_rest_url[0]}models/filter_options"
        LOGGER.info(f"Testing namedQueries in filter_options endpoint: {url}")

        response = execute_get_command(
            url=url,
            headers=get_rest_headers(token=user_token_for_api_calls),
        )

        named_queries = response.get("namedQueries", {})
        assert named_queries, "Named queries should be present in the response"

        default_perf_filters = named_queries.get(default_performance_filters, {})
        assert default_perf_filters, f"Named query '{default_performance_filters}' should be present"

        # Validate expected properties are present in the named query
        expected_properties = {
            "artifacts.requests_per_second.double_value",
            "artifacts.ttft_p90.double_value",
            "artifacts.use_case.string_value",
        }

        assert expected_properties == default_perf_filters.keys(), (
            f"default-performance-filters should contain exactly {expected_properties}, "
            f"but got {default_perf_filters.keys()}"
        )

        LOGGER.info("Named queries validation passed successfully")
