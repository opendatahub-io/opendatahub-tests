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
)
from tests.model_registry.model_catalog.constants import (
    REDHAT_AI_CATALOG_NAME,
    REDHAT_AI_VALIDATED_UNESCAPED_CATALOG_NAME,
    REDHAT_AI_CATALOG_ID,
    VALIDATED_CATALOG_ID,
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
            db_values = set(expected_properties[prop_name])
            api_values = set(api_filters[prop_name]["values"])

            missing_values = db_values - api_values
            extra_values = api_values - db_values

            if missing_values:
                error_msg = (
                    f"Property '{prop_name}': DB has {len(missing_values)} values missing from API: {missing_values}"
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
            error_msg = f"Property '{prop_name}': In API ({len(api_filters[prop_name]['values'])} values) but NOT in DB"
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
    errors = []

    # Get expected results from database (returns strings)
    expected_model_ids = set(get_models_matching_search_from_database(search_term, namespace, source_label))
    filter_desc = f"'{search_term}'" + (f" with source_label='{source_label}'" if source_label else "")
    LOGGER.info(f"Database query found {len(expected_model_ids)} models for {filter_desc}")

    # Get actual results from API (returns strings)
    api_models = api_response.get("items", [])
    actual_model_ids = set(model.get("id") for model in api_models if model.get("id"))
    LOGGER.info(f"API returned {len(actual_model_ids)} models for {filter_desc}")

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


def get_models_from_catalog_api(
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    page_size: int = 100,
    source_label: str | None = None,
    q: str | None = None,
    additional_params: str = "",
) -> dict[str, Any]:
    """
    Helper method to get models from catalog API with optional filtering

    Args:
        model_catalog_rest_url: REST URL for model catalog
        model_registry_rest_headers: Headers for model registry REST API
        page_size: Number of results per page
        source_label: Source label(s) to filter by (must be comma-separated for multiple filters)
        q: Free-form keyword search to filter models
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
