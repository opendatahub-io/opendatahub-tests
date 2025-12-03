from typing import Any, Tuple, List
import yaml

from kubernetes.dynamic import DynamicClient
from simple_logger.logger import get_logger


from ocp_resources.pod import Pod
from ocp_resources.config_map import ConfigMap
from tests.model_registry.model_catalog.constants import DEFAULT_CATALOGS
from tests.model_registry.model_catalog.db_constants import (
    SEARCH_MODELS_DB_QUERY,
    SEARCH_MODELS_WITH_SOURCE_ID_DB_QUERY,
    FILTER_MODELS_BY_LICENSE_DB_QUERY,
    FILTER_MODELS_BY_LICENSE_AND_LANGUAGE_DB_QUERY,
)
from tests.model_registry.model_catalog.constants import (
    REDHAT_AI_CATALOG_NAME,
    REDHAT_AI_VALIDATED_UNESCAPED_CATALOG_NAME,
    REDHAT_AI_CATALOG_ID,
    VALIDATED_CATALOG_ID,
    CATALOG_CONTAINER,
    PERFORMANCE_DATA_DIR,
)
from tests.model_registry.utils import execute_get_command

LOGGER = get_logger(name=__name__)


class ResourceNotFoundError(Exception):
    pass


def validate_model_catalog_enabled(pod: Pod) -> bool:
    for container in pod.instance.spec.containers:
        for env in container.env:
            if env.name == "ENABLE_MODEL_CATALOG":
                return True
    return False


def validate_model_catalog_resource(
    kind: Any, admin_client: DynamicClient, namespace: str, expected_resource_count: int
) -> None:
    resource = list(kind.get(namespace=namespace, label_selector="component=model-catalog", dyn_client=admin_client))
    assert resource
    LOGGER.info(f"Validating resource: {kind}: Found {len(resource)})")
    assert len(resource) == expected_resource_count, (
        f"Unexpected number of {kind} resources found: {[res.name for res in resource]}"
    )


def validate_default_catalog(catalogs: list[dict[Any, Any]]) -> None:
    errors = []
    for catalog in catalogs:
        expected_catalog = DEFAULT_CATALOGS.get(catalog["id"])
        assert expected_catalog, f"Unexpected catalog: {catalog}"
        for key, expected_value in expected_catalog.items():
            actual_value = catalog.get(key)
            if actual_value != expected_value:
                errors.append(f"For catalog '{catalog['id']}': expected {key}={expected_value}, but got {actual_value}")

    assert not errors, "\n".join(errors)


def get_validate_default_model_catalog_source(catalogs: list[dict[Any, Any]]) -> None:
    assert len(catalogs) == 2, f"Expected no custom models to be present. Actual: {catalogs}"
    ids_actual = [entry["id"] for entry in catalogs]
    assert sorted(ids_actual) == sorted(DEFAULT_CATALOGS.keys()), (
        f"Actual default catalog entries: {ids_actual},Expected: {DEFAULT_CATALOGS.keys()}"
    )


def extract_schema_fields(openapi_schema: dict[Any, Any], schema_name: str) -> tuple[set[str], set[str]]:
    """
    Extract all and required fields from an OpenAPI schema for validation.

    Args:
        openapi_schema: The parsed OpenAPI schema dictionary
        schema_name: Name of the schema to extract (e.g., "CatalogModel", "CatalogModelArtifact")

    Returns:
        Tuple of (all_fields, required_fields) excluding server-generated fields and timestamps.
    """

    def _extract_properties_and_required(schema: dict[Any, Any]) -> tuple[set[str], set[str]]:
        """Recursively extract properties and required fields from a schema."""
        props = set(schema.get("properties", {}).keys())
        required = set(schema.get("required", []))

        # Properties from allOf (inheritance/composition)
        if "allOf" in schema:
            for item in schema["allOf"]:
                sub_schema = item
                if "$ref" in item:
                    # Follow reference and recursively extract
                    ref_schema_name = item["$ref"].split("/")[-1]
                    sub_schema = openapi_schema["components"]["schemas"][ref_schema_name]
                sub_props, sub_required = _extract_properties_and_required(schema=sub_schema)
                props.update(sub_props)
                required.update(sub_required)

        return props, required

    target_schema = openapi_schema["components"]["schemas"][schema_name]
    all_properties, required_fields = _extract_properties_and_required(schema=target_schema)

    # Exclude fields that shouldn't be compared
    excluded_fields = {
        "id",  # Server-generated
        "externalId",  # Server-generated
        "createTimeSinceEpoch",  # Timestamps may differ
        "lastUpdateTimeSinceEpoch",  # Timestamps may differ
        "artifacts",  # CatalogModel only
        "source_id",  # CatalogModel only
    }

    return all_properties - excluded_fields, required_fields - excluded_fields


def validate_filter_options_structure(
    response: dict[Any, Any], expected_properties: set[str] | None = None
) -> Tuple[bool, List[str]]:
    """
    Comprehensive validation of filter_options response structure.

    Validates:
    - Top-level structure (filters object)
    - All property types and their required fields
    - Core properties presence (if specified)
    - String properties: type, values array, distinct values
    - Numeric properties: type, range object, min/max validity

    Args:
        response: The API response to validate
        expected_properties: Optional set of core properties that must be present

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Validate top-level structure
    if not isinstance(response, dict):
        errors.append("Response should be a dictionary")
        return False, errors

    if "filters" not in response:
        errors.append("Response should contain 'filters' object")
        return False, errors

    filters = response["filters"]
    if not isinstance(filters, dict):
        errors.append("Filters should be a dictionary")
        return False, errors

    if not filters:
        errors.append("Filters object should not be empty")
        return False, errors

    # Validate expected core properties if specified
    if expected_properties:
        for prop in expected_properties:
            if prop not in filters:
                errors.append(f"Core property '{prop}' should be present in filter options")

    # Validate each property structure
    for prop_name, prop_data in filters.items():
        if not isinstance(prop_data, dict):
            errors.append(f"Property '{prop_name}' should be a dictionary")
            continue

        if "type" not in prop_data:
            errors.append(f"Property '{prop_name}' should have 'type' field")
            continue

        prop_type = prop_data["type"]
        if not isinstance(prop_type, str) or not prop_type.strip():
            errors.append(f"Type for '{prop_name}' should be a non-empty string")
            continue

        # Validate string properties
        if prop_type == "string":
            if "values" not in prop_data:
                errors.append(f"String property '{prop_name}' should have 'values' array")
                continue

            values = prop_data["values"]
            if not isinstance(values, list):
                errors.append(f"Values for '{prop_name}' should be a list")
                continue

            if not values:
                errors.append(f"Values array for '{prop_name}' should not be empty")
                continue

            # Validate individual values
            for i, value in enumerate(values):
                if not isinstance(value, str):
                    errors.append(f"Value at index {i} for '{prop_name}' should be string, got: {type(value)}")
                elif not value.strip():
                    errors.append(f"Value at index {i} for '{prop_name}' should not be empty or whitespace")

            # Check for distinct values (no duplicates)
            try:
                if len(values) != len(set(values)):
                    errors.append(f"Values for '{prop_name}' should be distinct (found duplicates)")
            except TypeError:
                errors.append(f"Values for '{prop_name}' should be a list of strings, found unhashable type")

        # Validate numeric properties - checking multiple type names since we don't know what the API will return
        elif prop_type in ["number", "numeric", "float", "integer", "int"]:
            if "range" not in prop_data:
                errors.append(f"Numeric property '{prop_name}' should have 'range' object")
                continue

            range_obj = prop_data["range"]
            if not isinstance(range_obj, dict):
                errors.append(f"Range for '{prop_name}' should be a dictionary")
                continue

            # Check min/max presence
            if "min" not in range_obj:
                errors.append(f"Range for '{prop_name}' should have 'min' value")
            if "max" not in range_obj:
                errors.append(f"Range for '{prop_name}' should have 'max' value")

            if "min" in range_obj and "max" in range_obj:
                min_val = range_obj["min"]
                max_val = range_obj["max"]

                # Validate min/max are numeric
                if not isinstance(min_val, (int, float)):
                    errors.append(f"Min value for '{prop_name}' should be numeric, got: {type(min_val)}")
                if not isinstance(max_val, (int, float)):
                    errors.append(f"Max value for '{prop_name}' should be numeric, got: {type(max_val)}")

                # Validate logical relationship (min <= max)
                if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)) and min_val > max_val:
                    errors.append(f"Min value ({min_val}) should be <= max value ({max_val}) for '{prop_name}'")

    return len(errors) == 0, errors


def validate_model_catalog_configmap_data(configmap: ConfigMap, num_catalogs: int) -> None:
    """
    Validate the model catalog configmap data.

    Args:
        configmap: The ConfigMap object to validate
        num_catalogs: Expected number of catalogs in the configmap
    """
    # Check that model catalog configmaps is created when model registry is
    # enabled on data science cluster.
    catalogs = yaml.safe_load(configmap.instance.data["sources.yaml"])["catalogs"]
    assert len(catalogs) == num_catalogs, f"{configmap.name} should have {num_catalogs} catalog"
    if num_catalogs:
        validate_default_catalog(catalogs=catalogs)


def get_postgres_pod_in_namespace(namespace: str = "rhoai-model-registries") -> Pod:
    """Get the PostgreSQL pod for model catalog database."""
    postgres_pods = list(Pod.get(namespace=namespace, label_selector="app.kubernetes.io/name=model-catalog-postgres"))
    assert postgres_pods, f"No PostgreSQL pod found in namespace {namespace}"
    return postgres_pods[0]


def execute_database_query(query: str, namespace: str = "rhoai-model-registries") -> str:
    """
    Execute a SQL query against the model catalog database.

    Args:
        query: SQL query to execute
        namespace: OpenShift namespace containing the PostgreSQL pod

    Returns:
        Raw database query result as string
    """
    postgres_pod = get_postgres_pod_in_namespace(namespace=namespace)

    return postgres_pod.execute(
        command=["psql", "-U", "catalog_user", "-d", "model_catalog", "-c", query],
        container="postgresql",
    )


def parse_psql_output(psql_output: str) -> dict[str, Any]:
    """
    Parse psql CLI output into appropriate Python data structures.

    Handles two main formats:
    1. Single column: Returns {"values": [list_of_values]}
    2. Two columns with array_agg: Returns {"properties": {name: [values]}}

    Args:
        psql_output: Raw psql command output

    Returns:
        Dictionary with parsed data in appropriate format
    """
    lines = psql_output.strip().split("\n")

    # Find the header line to determine format
    header_line = None
    separator_line = None

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Look for separator line (all dashes and pipes)
        if line.replace("-", "").replace("+", "").replace("|", "").strip() == "":
            separator_line = i
            if i > 0:
                header_line = i - 1
            break

    if header_line is None:
        return {"values": []}

    header = lines[header_line].strip()

    # Determine format based on header
    if "|" in header and "array_agg" in header:
        # Two-column format with array aggregation
        return {"properties": _parse_array_agg_format(lines, separator_line + 1)}
    else:
        # Single column format
        return {"values": _parse_single_column_format(lines, separator_line + 1)}


def _parse_array_agg_format(lines: list[str], data_start: int) -> dict[str, list[str]]:
    """Parse two-column format with PostgreSQL array aggregation."""
    result = {}

    for line in lines[data_start:]:
        line = line.strip()
        if not line or "|" not in line:
            continue

        # Skip summary lines like "(X rows)"
        if line.startswith("(") and "row" in line:
            break

        # Parse data row: "property_name | {val1,val2,val3}"
        parts = line.split("|", 1)
        if len(parts) != 2:
            continue

        property_name = parts[0].strip()
        array_str = parts[1].strip()

        # Parse PostgreSQL array format: {val1,val2,val3}
        if array_str.startswith("{") and array_str.endswith("}"):
            # Remove braces and split by comma
            values_str = array_str[1:-1]
            if values_str:
                # Handle escaped commas and quotes properly
                values = [v.strip().strip('"') for v in values_str.split(",")]
                result[property_name] = values
            else:
                result[property_name] = []

    return result


def _parse_single_column_format(lines: list[str], data_start: int) -> list[str]:
    """Parse single column format."""
    result = []

    for line in lines[data_start:]:
        line = line.strip()
        if not line:
            continue

        # Skip summary lines like "(X rows)"
        if line.startswith("(") and "row" in line:
            break

        result.append(line)

    return result


def compare_filter_options_with_database(
    api_filters: dict[str, Any], db_properties: dict[str, list[str]], excluded_fields: set[str]
) -> Tuple[bool, List[str]]:
    """
    Compare API filter options response with database query results.

    Note: Currently assumes all properties are string types. Numeric/range
    properties are not returned by the API or DB query at this time.

    Args:
        api_filters: The "filters" dict from API response
        db_properties: Raw database properties before API filtering
        excluded_fields: Fields that API excludes from response

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    comparison_errors = []

    # Apply the same filtering logic the API uses
    expected_properties = {name: values for name, values in db_properties.items() if name not in excluded_fields}

    LOGGER.info(f"Database returned {len(db_properties)} total properties")
    LOGGER.info(
        f"After applying API filtering, expecting {len(expected_properties)} properties: {list(expected_properties.keys())}"  # noqa: E501
    )

    # Check for missing/extra properties
    missing_in_api = set(expected_properties.keys()) - set(api_filters.keys())
    extra_in_api = set(api_filters.keys()) - set(expected_properties.keys())

    # Log detailed comparison for each property
    for prop_name in sorted(set(expected_properties.keys()) | set(api_filters.keys())):
        if prop_name in expected_properties and prop_name in api_filters:
            db_data = expected_properties[prop_name]
            api_filter = api_filters[prop_name]

            # Check if this is a numeric property (has "range" in API response)
            if "range" in api_filter:
                # Numeric property: DB has [min, max] as 2-element array
                if len(db_data) == 2:
                    try:
                        db_min, db_max = float(db_data[0]), float(db_data[1])
                        api_min = api_filter["range"]["min"]
                        api_max = api_filter["range"]["max"]

                        if db_min != api_min or db_max != api_max:
                            error_msg = (
                                f"Property '{prop_name}': Range mismatch - DB: [{db_min}, {db_max}], "
                                f"API: [{api_min}, {api_max}]"
                            )
                            LOGGER.error(error_msg)
                            comparison_errors.append(error_msg)
                        else:
                            LOGGER.info(f"Property '{prop_name}': Perfect range match (min={api_min}, max={api_max})")
                    except (ValueError, TypeError) as e:
                        error_msg = f"Property '{prop_name}': Failed to parse numeric values - {e}"
                        LOGGER.error(error_msg)
                        comparison_errors.append(error_msg)
                else:
                    error_msg = f"Property '{prop_name}': Expected 2 values for range, got {len(db_data)}"
                    LOGGER.error(error_msg)
                    comparison_errors.append(error_msg)
            else:
                # String/array property: compare values as sets
                db_values = set(db_data)
                api_values = set(api_filter["values"])

                missing_values = db_values - api_values
                extra_values = api_values - db_values

                if missing_values:
                    error_msg = (
                        f"Property '{prop_name}': DB has {len(missing_values)} "
                        f"values missing from API: {missing_values}"
                    )
                    LOGGER.error(error_msg)
                    comparison_errors.append(error_msg)
                if extra_values:
                    error_msg = (
                        f"Property '{prop_name}': API has {len(extra_values)} values missing from DB: {extra_values}"
                    )
                    LOGGER.error(error_msg)
                    comparison_errors.append(error_msg)
                if not missing_values and not extra_values:
                    LOGGER.info(f"Property '{prop_name}': Perfect match ({len(api_values)} values)")
        elif prop_name in expected_properties:
            error_msg = f"Property '{prop_name}': In DB ({len(expected_properties[prop_name])} values) but NOT in API"
            LOGGER.error(error_msg)
            comparison_errors.append(error_msg)
        elif prop_name in api_filters:
            LOGGER.info(f"Property name: '{prop_name}' in API filters: {api_filters[prop_name]}")
            # For properties only in API, we can't reliably get DB values, so skip logging them
            if "range" in api_filters[prop_name]:
                error_msg = f"Property '{prop_name}': In API (range property) but NOT in DB"
            else:
                error_msg = (
                    f"Property '{prop_name}': In API ({len(api_filters[prop_name]['values'])} values) but NOT in DB"
                )
            LOGGER.error(error_msg)
            comparison_errors.append(error_msg)

    # Check for property-level mismatches
    if missing_in_api:
        comparison_errors.append(f"API missing properties found in database: {missing_in_api}")

    if extra_in_api:
        comparison_errors.append(f"API has extra properties not in database: {extra_in_api}")

    is_valid = len(comparison_errors) == 0
    return is_valid, comparison_errors


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
    search_term: str, namespace: str = "rhoai-model-registries", source_label: str | None = None
) -> list[str]:
    """
    Query the database directly to find model IDs that should match the search term.

    Uses SEARCH_MODELS_DB_QUERY from db_constants to replicate the exact backend search logic
    from applyCatalogModelListFilters function in kubeflow/model-registry.

    Args:
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
                f"Unknown source_label: '{source_label}'. Supported labels: {REDHAT_AI_CATALOG_NAME}, {REDHAT_AI_VALIDATED_UNESCAPED_CATALOG_NAME}"  # noqa: E501
            )

        # Use the extended query with source_id filtering from db_constants
        search_query = SEARCH_MODELS_WITH_SOURCE_ID_DB_QUERY.format(
            search_pattern=search_pattern, source_ids=f"'{catalog_id}'"
        )
    else:
        # Use the standardized search query from db_constants
        search_query = SEARCH_MODELS_DB_QUERY.format(search_pattern=search_pattern)

    db_result = execute_database_query(query=search_query, namespace=namespace)
    parsed_result = parse_psql_output(psql_output=db_result)

    return parsed_result.get("values", [])


def get_models_matching_filter_query_from_database(
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
    db_result = execute_database_query(query=filter_query_sql, namespace=namespace)
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
    actual_model_ids = set(model.get("id") for model in api_models if model.get("id"))
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
    api_response: dict[str, Any],
    search_term: str,
    namespace: str = "rhoai-model-registries",
    source_label: str | None = None,
) -> tuple[bool, list[str]]:
    """
    Validate API search results against database query results.

    Args:
        api_response: API response from search query
        search_term: Search term used
        namespace: OpenShift namespace for PostgreSQL pod
        source_label: Optional source label filter used in the API call

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    # Get expected results from database
    expected_model_ids = set(get_models_matching_search_from_database(search_term, namespace, source_label))
    filter_desc = f"search term '{search_term}'" + (f" with source_label='{source_label}'" if source_label else "")
    LOGGER.info(f"Database query found {len(expected_model_ids)} models for {filter_desc}")

    # Compare with API results
    return _compare_api_and_database_results(
        api_response=api_response, expected_model_ids=expected_model_ids, description=filter_desc
    )


def validate_filter_query_results_against_database(
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


def get_models_from_catalog_api(
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    page_size: int = 100,
    source_label: str | None = None,
    q: str | None = None,
    order_by: str | None = None,
    sort_order: str | None = None,
    additional_params: str = "",
) -> dict[str, Any]:
    """
    Helper method to get models from catalog API with optional filtering and sorting

    Args:
        model_catalog_rest_url: REST URL for model catalog
        model_registry_rest_headers: Headers for model registry REST API
        page_size: Number of results per page
        source_label: Source label(s) to filter by (must be comma-separated for multiple filters)
        q: Free-form keyword search to filter models
        order_by: Field to order results by (ID, NAME, CREATE_TIME, LAST_UPDATE_TIME)
        sort_order: Sort order (ASC or DESC)
        additional_params: Additional query parameters (e.g., "&filterQuery=name='model_name'")

    Returns:
        Dictionary containing the API response
    """
    base_url = f"{model_catalog_rest_url[0]}models"

    # Build params dictionary for proper URL encoding
    params = {"pageSize": page_size}

    if source_label:
        params["sourceLabel"] = source_label

    if q:
        params["q"] = q

    if order_by:
        params["orderBy"] = order_by

    if sort_order:
        params["sortOrder"] = sort_order

    # Parse additional_params string into params dict for proper URL encoding
    if additional_params:
        # Remove leading & if present
        clean_params = additional_params.lstrip("&")
        # Split by & and then by = to get key-value pairs
        for param in clean_params.split("&"):
            if "=" in param:
                key, value = param.split("=", 1)  # Split only on first = to handle values with =
                params[key] = value

    return execute_get_command(url=base_url, headers=model_registry_rest_headers, params=params)


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


def get_sources_with_sorting(
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    order_by: str,
    sort_order: str,
) -> dict[str, Any]:
    """
    Get sources with sorting parameters

    Args:
        model_catalog_rest_url: REST URL for model catalog
        model_registry_rest_headers: Headers for model registry REST API
        order_by: Field to order results by (ID, NAME)
        sort_order: Sort order (ASC or DESC)

    Returns:
        Dictionary containing the API response
    """
    base_url = f"{model_catalog_rest_url[0]}sources"
    params = {
        "orderBy": order_by,
        "sortOrder": sort_order,
        "pageSize": 100,
    }

    return execute_get_command(url=base_url, headers=model_registry_rest_headers, params=params)


def get_artifacts_with_sorting(
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    source_id: str,
    model_name: str,
    order_by: str,
    sort_order: str,
) -> dict[str, Any]:
    """
    Get artifacts with sorting parameters

    Args:
        model_catalog_rest_url: REST URL for model catalog
        model_registry_rest_headers: Headers for model registry REST API
        source_id: Source ID for the model
        model_name: Name of the model
        order_by: Field to order results by
        sort_order: Sort order (ASC or DESC)

    Returns:
        Dictionary containing the API response
    """
    base_url = f"{model_catalog_rest_url[0]}sources/{source_id}/models/{model_name}/artifacts"
    params = {
        "orderBy": order_by,
        "sortOrder": sort_order,
        "pageSize": 100,
    }

    return execute_get_command(url=base_url, headers=model_registry_rest_headers, params=params)


def validate_items_sorted_correctly(items: list[dict], field: str, order: str) -> bool:
    """
    Extract field values and verify they're in correct order

    Args:
        items: List of items to validate
        field: Field name to check sorting on
        order: Sort order (ASC or DESC)

    Returns:
        True if items are sorted correctly, False otherwise
    """
    if len(items) <= 1:
        if field == "NAME" and items[0].get("artifactType") == "model-artifact":
            # When testing sorting for model artifacts we use only models from the validated catalog, since
            # they almost all have more than 1 artifact. However, some of these models still return a single artifact.
            # Given that this is currently the expected behavior, we return True.
            single_artifact_models = [
                "mistral-small-24B",
                "gemma-2",
                "granite-3.1-8b-base-quantized.w4a16",
                "granite-3.1-8b-instruct-FP8-dynamic",
                "granite-3.1-8b-starter-v2",
            ]
            if any(single_artifact_model in items[0].get("uri") for single_artifact_model in single_artifact_models):
                return True
        else:
            # In any other case, we expect at least 2 items to sort.
            raise ValueError(f"At least 2 items are required to sort, got {len(items)}")

    # Extract field values for comparison
    values = []
    for item in items:
        if field == "ID":
            value = item.get("id")
        elif field == "NAME":
            value = item.get("name")
        elif field == "CREATE_TIME":
            value = item.get("createTimeSinceEpoch")
        elif field == "LAST_UPDATE_TIME":
            value = item.get("lastUpdateTimeSinceEpoch")
        else:
            raise ValueError(f"Invalid field: {field}")

        if value is None:
            raise ValueError(f"Field {field} is missing from item: {item}")

        values.append(value)

    # Convert values to appropriate types for comparison
    if field == "ID":
        # Convert IDs to integers for numeric comparison
        try:
            values = [int(v) for v in values]
        except ValueError:
            # If conversion fails, fall back to string comparison
            values = [str(v) for v in values]

    # Check if values are in correct order
    if order == "ASC":
        return all(values[i] <= values[i + 1] for i in range(len(values) - 1))
    elif order == "DESC":
        return all(values[i] >= values[i + 1] for i in range(len(values) - 1))
    else:
        raise ValueError(f"Invalid sort order: {order}")


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
