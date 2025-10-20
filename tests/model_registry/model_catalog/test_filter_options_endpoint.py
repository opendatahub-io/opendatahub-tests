import pytest
from typing import Self
from simple_logger.logger import get_logger

from tests.model_registry.model_catalog.utils import execute_get_command, validate_filter_options_structure
from tests.model_registry.utils import get_rest_headers
from utilities.user_utils import UserTestSession

LOGGER = get_logger(name=__name__)

pytestmark = [
    pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace", "original_user")
]


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
class TestFilterOptionsEndpoint:
    """
    Test class for validating the models/filter_options endpoint
    RHOAIENG-36696
    """

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

        # Log details about the validated properties
        for prop_name, prop_data in filters.items():
            if prop_data.get("type") == "string" and "values" in prop_data:
                values_count = len(prop_data["values"])
                LOGGER.info(f"String property '{prop_name}' validated with {values_count} distinct values")
            elif prop_data.get("type") in ["number", "numeric", "float", "integer", "int"] and "range" in prop_data:
                range_obj = prop_data["range"]
                LOGGER.info(
                    f"Numeric property '{prop_name}' validated: {range_obj.get('min')} to {range_obj.get('max')}"
                )

        LOGGER.info("All filter options validation passed successfully")

    @pytest.mark.skip(reason="TODO: Implement after investigating backend DB queries")
    def test_comprehensive_coverage_against_database(
        self: Self,
        model_catalog_rest_url: list[str],
        user_token_for_api_calls: str,
        test_idp_user: UserTestSession,
    ):
        """
        STUBBED: Validate filter options are comprehensive across all sources/models in DB.
        Acceptance Criteria: The returned options are comprehensive and not limited to a
        subset of models or a single source.

        TODO IMPLEMENTATION PLAN:
        1. Investigate backend endpoint logic:
           - Find the source code for /models/filter_options endpoint in kubeflow/model-registry
           - Understand what DB tables it queries (likely model/artifact tables)
           - Identify the exact SQL queries used to build filter values
           - Determine database schema and column names

        2. Replicate queries via pod shell:
           - Use get_model_catalog_pod() to access catalog pod
           - Execute psql commands via pod.execute()
           - Query same tables/columns the endpoint uses
           - Extract all distinct values for string properties: SELECT DISTINCT license FROM models;
           - Extract min/max ranges for numeric properties: SELECT MIN(metric), MAX(metric) FROM models;

        3. Compare results:
           - API response filter values should match DB query results exactly
           - Ensure no values are missing (comprehensive coverage)
           - Validate across all sources, not just one

        4. DB Access Pattern Example:
           catalog_pod = get_model_catalog_pod(client, namespace)[0]
           result = catalog_pod.execute(
               command=["psql", "-U", "catalog_user", "-d", "catalog_db", "-c", "SELECT DISTINCT license FROM models;"],
               container="catalog"
           )

        5. Implementation considerations:
           - Handle different data types (strings vs arrays like tasks)
           - Parse psql output correctly
           - Handle null/empty values
           - Ensure database connection credentials are available
        """
        pytest.skip("TODO: Implement comprehensive coverage validation after backend investigation")
