import pytest
from typing import Self
from simple_logger.logger import get_logger

from tests.model_registry.model_catalog.utils import (
    validate_filter_options_structure,
    execute_database_query,
    parse_psql_output,
    compare_filter_options_with_database,
)
from tests.model_registry.model_catalog.db_constants import FILTER_OPTIONS_DB_QUERY, API_EXCLUDED_FILTER_FIELDS
from tests.model_registry.utils import get_rest_headers, execute_get_command
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

    @pytest.mark.parametrize(
        "user_token_for_api_calls,",
        [
            pytest.param(
                {},
                id="test_filter_options_admin_user",
            ),
            pytest.param(
                {"user_type": "test"},
                id="test_filter_options_non_admin_user",
            ),
            pytest.param(
                {"user_type": "sa_user"},
                id="test_filter_options_service_account",
            ),
        ],
        indirect=["user_token_for_api_calls"],
    )
    def test_filter_options_endpoint_validation(
        self: Self,
        model_catalog_rest_url: list[str],
        user_token_for_api_calls: str,
        test_idp_user: UserTestSession,
    ):
        """
        Comprehensive test for filter_options endpoint.
        Validates all acceptance criteria:
        - A GET request returns a 200 OK response
        - Response includes filter options for string-based properties with values array containing distinct values
        - Response includes filter options for numeric properties with range object containing min/max values
        - Core properties are present (license, provider, tasks, validated_on)
        """
        url = f"{model_catalog_rest_url[0]}models/filter_options"
        LOGGER.info(f"Testing filter_options endpoint: {url}")

        # This will raise an exception if the status code is not 200/201 (validates acceptance criteria #1)
        response = execute_get_command(
            url=url,
            headers=get_rest_headers(token=user_token_for_api_calls),
        )

        assert response is not None, "Filter options response should not be None"
        LOGGER.info("Filter options endpoint successfully returned 200 OK")

        # Expected core properties based on current API response
        expected_properties = {"license", "provider", "tasks", "validated_on"}

        # Comprehensive validation using single function (validates acceptance criteria #2, #3, #4)
        is_valid, errors = validate_filter_options_structure(response=response, expected_properties=expected_properties)
        assert is_valid, f"Filter options validation failed: {'; '.join(errors)}"

        filters = response["filters"]
        LOGGER.info(f"Found {len(filters)} filter properties: {list(filters.keys())}")
        LOGGER.info("All filter options validation passed successfully")

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
    @pytest.mark.xfail(strict=True, reason="RHOAIENG-37069: backend/API discrepancy expected")
    def test_comprehensive_coverage_against_database(
        self: Self,
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

        db_result = execute_database_query(query=FILTER_OPTIONS_DB_QUERY, namespace=model_registry_namespace)
        parsed_result = parse_psql_output(psql_output=db_result)

        db_properties = parsed_result.get("properties", {})
        LOGGER.info(f"Raw database query returned {len(db_properties)} properties: {list(db_properties.keys())}")

        is_valid, comparison_errors = compare_filter_options_with_database(
            api_filters=api_filters, db_properties=db_properties, excluded_fields=API_EXCLUDED_FILTER_FIELDS
        )

        if not is_valid:
            failure_msg = "Filter options API response does not match database content"
            failure_msg += "\nDetailed comparison errors:\n" + "\n".join(comparison_errors)
            assert False, failure_msg

        LOGGER.info("Comprehensive database coverage validation passed - API matches database exactly")
