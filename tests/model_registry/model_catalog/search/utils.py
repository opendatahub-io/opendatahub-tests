"""Utility functions for model catalog search tests."""

from typing import Any

from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger

from tests.model_registry.model_catalog.constants import (
    CATALOG_CONTAINER,
    PERFORMANCE_DATA_DIR,
    REDHAT_AI_CATALOG_ID,
    REDHAT_AI_CATALOG_NAME,
    REDHAT_AI_VALIDATED_UNESCAPED_CATALOG_NAME,
    VALIDATED_CATALOG_ID,
)
from tests.model_registry.model_catalog.db_constants import (
    FILTER_MODELS_BY_LICENSE_AND_LANGUAGE_DB_QUERY,
    FILTER_MODELS_BY_LICENSE_DB_QUERY,
    SEARCH_MODELS_DB_QUERY,
    SEARCH_MODELS_WITH_SOURCE_ID_DB_QUERY,
)
from tests.model_registry.model_catalog.utils import execute_database_query, parse_psql_output
from tests.model_registry.utils import execute_get_command

LOGGER = get_logger(name=__name__)


def validate_model_contains_search_term(model: dict[str, Any], search_term: str) -> bool:
    """
    Verify model contains search term in searchable fields based on backend implementation.

    Searchable fields: name, description, provider, libraryName, tasks

    Args:
        model: Model dictionary from API response
        search_term: Search term to validate

    Returns:
        True if model contains search term in any searchable field
    """
    search_term_lower = search_term.lower()

    searchable_content = [
        model.get("name", "").lower(),
        model.get("description", "").lower(),
        model.get("provider", "").lower(),
        model.get("libraryName", "").lower(),
        " ".join(model.get("tasks", [])).lower() if model.get("tasks") else "",
    ]

    return any(search_term_lower in content for content in searchable_content if content)


def get_models_matching_search_from_database(
    admin_client: DynamicClient,
    search_term: str,
    namespace: str = "rhoai-model-registries",
    source_label: str | None = None,
) -> list[str]:
    """
    Query the database directly to find model IDs that should match the search term.

    Uses SEARCH_MODELS_DB_QUERY from db_constants to replicate the exact backend search logic
    from applyCatalogModelListFilters function in kubeflow/model-registry.

    Args:
        admin_client: DynamicClient to connect to database
        search_term: Search term to find
        namespace: OpenShift namespace containing the PostgreSQL pod
        source_label: Optional source label to filter by (e.g., "Red+Hat+AI")

    Returns:
        List of model IDs that contain the search term in searchable fields and match source filter
    """

    # Escape single quotes and create the search pattern: %%%s%%
    escaped_term = search_term.replace("'", "''")
    search_pattern = f"%{escaped_term.lower()}%"

    # Choose query based on whether source filtering is needed
    if source_label:
        # Simple direct mapping check
        if source_label == REDHAT_AI_CATALOG_NAME:
            catalog_id = REDHAT_AI_CATALOG_ID
        elif source_label == REDHAT_AI_VALIDATED_UNESCAPED_CATALOG_NAME:
            catalog_id = VALIDATED_CATALOG_ID
        else:
            raise ValueError(
                f"Unknown source_label: '{source_label}'. "
                f"Supported labels: {REDHAT_AI_CATALOG_NAME}, {REDHAT_AI_VALIDATED_UNESCAPED_CATALOG_NAME}"
            )

        # Use the extended query with source_id filtering from db_constants
        search_query = SEARCH_MODELS_WITH_SOURCE_ID_DB_QUERY.format(
            search_pattern=search_pattern, source_ids=f"'{catalog_id}'"
        )
    else:
        # Use the standardized search query from db_constants
        search_query = SEARCH_MODELS_DB_QUERY.format(search_pattern=search_pattern)

    db_result = execute_database_query(admin_client=admin_client, query=search_query, namespace=namespace)
    parsed_result = parse_psql_output(psql_output=db_result)

    return parsed_result.get("values", [])


def get_models_matching_filter_query_from_database(
    admin_client: DynamicClient,
    licenses: str,
    language_pattern_1: str | None = None,
    language_pattern_2: str | None = None,
    namespace: str = "rhoai-model-registries",
) -> list[str]:
    """
    Query the database directly to find model IDs that match the filter criteria.

    Uses either FILTER_MODELS_BY_LICENSE_DB_QUERY or FILTER_MODELS_BY_LICENSE_AND_LANGUAGE_DB_QUERY
    from db_constants to replicate the exact backend filter query logic.

    Args:
        admin_client: DynamicClient to connect to database
        licenses: License values in SQL IN clause format (e.g., "'gemma','modified-mit'")
        language_pattern_1: First language pattern for ILIKE (e.g., '%it%'). Optional.
        language_pattern_2: Second language pattern for ILIKE (e.g., '%de%'). Optional.
        namespace: OpenShift namespace containing the PostgreSQL pod

    Returns:
        List of model IDs that match the filter criteria
    """
    # Select the appropriate query template based on whether language filters are provided
    if language_pattern_1 and language_pattern_2:
        filter_query_sql = FILTER_MODELS_BY_LICENSE_AND_LANGUAGE_DB_QUERY.format(
            licenses=licenses,
            language_pattern_1=language_pattern_1,
            language_pattern_2=language_pattern_2,
        )
    else:
        filter_query_sql = FILTER_MODELS_BY_LICENSE_DB_QUERY.format(licenses=licenses)

    LOGGER.debug(f"Filter query (SQL): {filter_query_sql}")

    # Execute the database query
    db_result = execute_database_query(admin_client=admin_client, query=filter_query_sql, namespace=namespace)
    parsed_result = parse_psql_output(psql_output=db_result)

    return parsed_result.get("values", [])


def _compare_api_and_database_results(
    api_response: dict[str, Any],
    expected_model_ids: set[str],
    description: str,
) -> tuple[bool, list[str]]:
    """
    Compare API response model IDs with expected database model IDs.

    Args:
        api_response: API response containing model items
        expected_model_ids: Set of model IDs expected from database
        description: Description of the query for logging (e.g., "search term 'granite'", "filter query")

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []

    # Get actual results from API
    api_models = api_response.get("items", [])
    actual_model_ids = {model.get("id") for model in api_models if model.get("id")}
    LOGGER.info(f"API returned {len(actual_model_ids)} models for {description}")

    # Compare results
    missing_in_api = expected_model_ids - actual_model_ids
    extra_in_api = actual_model_ids - expected_model_ids

    if missing_in_api:
        errors.append(f"API missing {len(missing_in_api)} models found in database: {missing_in_api}")

    if extra_in_api:
        errors.append(f"API returned {len(extra_in_api)} extra models not found in database: {extra_in_api}")

    # Log detailed comparison
    if expected_model_ids == actual_model_ids:
        LOGGER.info(f"Perfect match: API and database both found {len(expected_model_ids)} models")
    else:
        LOGGER.error(f"Mismatch: DB={len(expected_model_ids)}, API={len(actual_model_ids)}")

    return len(errors) == 0, errors


def validate_search_results_against_database(
    admin_client: DynamicClient,
    api_response: dict[str, Any],
    search_term: str,
    namespace: str = "rhoai-model-registries",
    source_label: str | None = None,
) -> tuple[bool, list[str]]:
    """
    Validate API search results against database query results.

    Args:
        admin_client: Admin client to use
        api_response: API response from search query
        search_term: Search term used
        namespace: OpenShift namespace for PostgreSQL pod
        source_label: Optional source label filter used in the API call

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    # Get expected results from database
    expected_model_ids = set(
        get_models_matching_search_from_database(
            admin_client=admin_client, search_term=search_term, namespace=namespace, source_label=source_label
        )
    )
    filter_desc = f"search term '{search_term}'" + (f" with source_label='{source_label}'" if source_label else "")
    LOGGER.info(f"Database query found {len(expected_model_ids)} models for {filter_desc}")

    # Compare with API results
    return _compare_api_and_database_results(
        api_response=api_response, expected_model_ids=expected_model_ids, description=filter_desc
    )


def validate_filter_query_results_against_database(
    admin_client: DynamicClient,
    api_response: dict[str, Any],
    licenses: str,
    language_pattern_1: str | None = None,
    language_pattern_2: str | None = None,
    namespace: str = "rhoai-model-registries",
) -> tuple[bool, list[str]]:
    """
    Validate API filter query results against database query results.

    Supports validation of filter queries with:
    - License filter only: license IN (...)
    - License and language filters: license IN (...) AND (language ILIKE ... OR language ILIKE ...)

    Args:
        admin_client: Admin client to use
        api_response: API response from filter query
        licenses: License values in SQL IN clause format (e.g., "'gemma','modified-mit'")
        language_pattern_1: First language pattern for ILIKE (e.g., '%it%'). Optional.
        language_pattern_2: Second language pattern for ILIKE (e.g., '%de%'). Optional.
        namespace: OpenShift namespace for PostgreSQL pod

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    # Get expected results from database
    expected_model_ids = set(
        get_models_matching_filter_query_from_database(
            admin_client=admin_client,
            licenses=licenses,
            language_pattern_1=language_pattern_1,
            language_pattern_2=language_pattern_2,
            namespace=namespace,
        )
    )

    # Build filter description based on whether language patterns are provided
    if language_pattern_1 and language_pattern_2:
        filter_desc = f"licenses IN ({licenses}) AND (language ILIKE '{language_pattern_1}' \
            OR language ILIKE '{language_pattern_2}')"
    else:
        filter_desc = f"licenses IN ({licenses})"

    LOGGER.info(f"Database query found {len(expected_model_ids)} models for filter: {filter_desc}")

    # Compare with API results
    return _compare_api_and_database_results(
        api_response=api_response, expected_model_ids=expected_model_ids, description=filter_desc
    )


def fetch_all_artifacts_with_dynamic_paging(
    url_with_pagesize: str,
    headers: dict[str, str],
    page_size: int = 100,
    page_size_increment: int = 50,
) -> dict[str, Any]:
    """
    Fetch all artifacts from an endpoint with dynamic page size adjustment.

    If pagination is detected (nextPageToken present), automatically increases
    page size and retries until all items fit in a single page.

    Args:
        url_with_pagesize: The paginated URL with pageSize parameter
        headers: Request headers
        page_size: Starting page size (default: 100)
        page_size_increment: Amount to increase page size on each retry (default: 50)

    Returns:
        The complete API response with all items in a single page
    """

    while True:
        paginated_url = f"{url_with_pagesize}={page_size}"
        response = execute_get_command(url=paginated_url, headers=headers)

        if not response.get("nextPageToken"):
            LOGGER.info(f"Fetched {len(response.get('items', []))} items with pageSize={page_size}")
            return response

        LOGGER.info(f"Pagination detected with pageSize={page_size}, increasing by {page_size_increment}")
        page_size += page_size_increment


def validate_performance_data_files_on_pod(model_catalog_pod: Pod) -> dict[str, list[str]]:
    """
    Validate that performance data files exist for all models in the catalog pod.

    Iterates through providers and models in the performance data directory to check
    for required metadata and performance files.

    Args:
        model_catalog_pod: Pod object for the model catalog pod

    Returns:
        Dictionary with validation results if missing files are found,
        Returns empty dictionary if all models have all required files.
    """
    validation_results = {}

    providers = model_catalog_pod.execute(container=CATALOG_CONTAINER, command=["ls", PERFORMANCE_DATA_DIR])

    for provider in providers.splitlines():
        required_files = ["metadata.json", "performance.ndjson", "evaluations.ndjson"]
        if provider == "manifest.json":
            continue
        LOGGER.info(f"Checking provider: {provider}")
        # Only for RedHatAI model we expect performance.ndjson file, based on edge case definition
        # https://docs.google.com/document/d/1K6SQi7Se8zljfB0UvXKKqV8VWVh5Pfq4HqKPtNvIQzg/edit?tab=t.0#heading=h.rh09auvgvlxd
        if provider != "RedHatAI":
            required_files.remove("performance.ndjson")
        models = model_catalog_pod.execute(
            container=CATALOG_CONTAINER, command=["ls", f"{PERFORMANCE_DATA_DIR}/{provider}"]
        )

        for model in models.splitlines():
            if model == "provider.json":
                continue
            # Remove data for specific RH models based on
            # https://redhat-internal.slack.com/archives/C09570S9VV0/p1762164394782969?thread_ts=1761834621.645019&cid=C09570S9VV0
            if model == "Mistral-Small-24B-Instruct-2501":
                required_files.remove("evaluations.ndjson")
            elif model == "granite-3.1-8b-instruct-quantized.w8a8":
                required_files.remove("performance.ndjson")

            result = model_catalog_pod.execute(
                container=CATALOG_CONTAINER, command=["ls", f"{PERFORMANCE_DATA_DIR}/{provider}/{model}"]
            )

            # Check which required files are missing
            missing_files = [f for f in required_files if f not in result]

            if missing_files:
                model_key = f"{provider}/{model}"
                validation_results[model_key] = missing_files

    if not validation_results:
        LOGGER.info("All models have all required performance data files on catalog pod")
    else:
        LOGGER.warning(f"Found models with missing performance data files: {validation_results}")

    return validation_results


def _validate_single_criterion(
    artifact_name: str, custom_properties: dict[str, Any], validation: dict[str, Any]
) -> tuple[bool, str]:
    """
    Helper function to validate a single criterion against an artifact.

    Args:
        artifact_name: Name of the artifact being validated
        custom_properties: Custom properties dictionary from the artifact
        validation: Single validation criterion containing key_name, key_type, comparison, value

    Returns:
        tuple: (condition_met: bool, message: str)
    """
    key_name = validation["key_name"]
    key_type = validation["key_type"]
    comparison_type = validation["comparison"]
    expected_val = validation["value"]

    raw_value = custom_properties.get(key_name, {}).get(key_type, None)

    if raw_value is None:
        return False, f"{key_name}: missing"

    # Convert value to appropriate type
    try:
        if key_type == "int_value":
            artifact_value = int(raw_value)
        elif key_type == "double_value":
            artifact_value = float(raw_value)
        elif key_type == "string_value":
            artifact_value = str(raw_value)
        else:
            LOGGER.warning(f"Unknown key_type: {key_type}")
            return False, f"{key_name}: unknown type {key_type}"
    except (ValueError, TypeError):
        return False, f"{key_name}: conversion error"

    # Perform comparison based on type
    condition_met = False
    if comparison_type == "exact":
        condition_met = artifact_value == expected_val
    elif comparison_type == "min":
        condition_met = artifact_value >= expected_val
    elif comparison_type == "max":
        condition_met = artifact_value <= expected_val
    elif comparison_type == "contains" and key_type == "string_value":
        condition_met = expected_val in artifact_value

    message = f"Artifact {artifact_name} {key_name}: {artifact_value} {comparison_type} {expected_val}"
    return condition_met, message


def _get_artifact_validation_results(
    artifact: dict[str, Any], expected_validations: list[dict[str, Any]]
) -> tuple[list[bool], list[str]]:
    """
    Checks one artifact against all validations and returns the boolean outcomes and messages.
    """
    artifact_name = artifact.get("name", "missing_artifact_name")
    custom_properties = artifact["customProperties"]

    # Store the boolean results and informative messages
    bool_results = []
    messages = []

    for validation in expected_validations:
        condition_met, message = _validate_single_criterion(
            artifact_name=artifact_name, custom_properties=custom_properties, validation=validation
        )
        bool_results.append(condition_met)
        messages.append(message)

    return bool_results, messages


def validate_model_artifacts_match_criteria_and(
    all_model_artifacts: list[dict[str, Any]], expected_validations: list[dict[str, Any]], model_name: str
) -> bool:
    """
    Validates that at least one artifact in the model satisfies ALL expected validation criteria.
    """
    for artifact in all_model_artifacts:
        bool_results, messages = _get_artifact_validation_results(
            artifact=artifact, expected_validations=expected_validations
        )
        # If ALL results are True
        if all(bool_results):
            validation_results = [f"{message}: passed" for message in messages]
            LOGGER.info(
                f"Model {model_name} passed all {len(bool_results)} validations with artifact: {validation_results}"
            )
            return True

    return False


def validate_model_artifacts_match_criteria_or(
    all_model_artifacts: list[dict[str, Any]], expected_validations: list[dict[str, Any]], model_name: str
) -> bool:
    """
    Validates that at least one artifact in the model satisfies AT LEAST ONE of the expected validation criteria.
    """
    for artifact in all_model_artifacts:
        bool_results, messages = _get_artifact_validation_results(
            artifact=artifact, expected_validations=expected_validations
        )
        if any(bool_results):
            # Find the first passing message for logging
            LOGGER.info(f"Model {model_name} passed OR validation with artifact: {messages[bool_results.index(True)]}")
            return True

    LOGGER.error(f"Model {model_name} failed all OR validations")
    return False


def validate_recommendations_subset(
    full_artifacts: list[dict[str, Any]], recommendations_artifacts: list[dict[str, Any]], model_name: str
) -> bool:
    """
    Validate that recommendations artifacts are a proper subset of all artifacts.

    Args:
        full_artifacts: All performance artifacts (recommendations=false)
        recommendations_artifacts: Filtered artifacts (recommendations=true)
        model_name: Model name for logging

    Returns:
        bool: True if validation passes

    Raises:
        AssertionError: If validation fails with descriptive message
    """
    LOGGER.info(f"Validating recommendations subset for model '{model_name}'")

    # Convert artifacts to comparable format (using artifact ID for comparison)
    full_artifact_ids = {artifact.get("id") for artifact in full_artifacts if artifact.get("id")}
    recommendations_artifact_ids = {artifact.get("id") for artifact in recommendations_artifacts if artifact.get("id")}

    # Check subset relationship: all recommendation IDs should exist in full results
    missing_in_full = recommendations_artifact_ids - full_artifact_ids
    if missing_in_full:
        error_msg = (
            f"Model '{model_name}': Found {len(missing_in_full)} recommendation artifacts "
            f"that don't exist in full results: {missing_in_full}"
        )
        LOGGER.error(error_msg)
        raise AssertionError(error_msg)

    # Log success details
    subset_percentage = (len(recommendations_artifacts) / len(full_artifacts)) * 100
    LOGGER.info(
        f"Model '{model_name}': Recommendations validation passed - "
        f"{len(recommendations_artifacts)}/{len(full_artifacts)} artifacts "
        f"({subset_percentage:.1f}% of total)"
    )

    return True
