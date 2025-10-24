from typing import Any, Tuple, List
import yaml

from kubernetes.dynamic import DynamicClient
from simple_logger.logger import get_logger


from ocp_resources.pod import Pod
from ocp_resources.config_map import ConfigMap
from tests.model_registry.model_catalog.constants import DEFAULT_CATALOGS
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


def parse_psql_array_agg_output(psql_output: str) -> dict[str, list[str]]:
    """
    Parse psql output from array_agg query into Python dict.

    Expected format:
        name     | array_agg
        ---------+----------
        license  | {apache-2.0,mit,bsd}
        provider | {Meta,Microsoft}

    Returns:
        dict mapping property names to lists of values
    """
    result = {}
    lines = psql_output.strip().split("\n")

    # Skip header lines (first 2-3 lines are typically headers and separators)
    data_started = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith("-") or "|" not in line:
            continue

        # Skip header row
        if "array_agg" in line and not data_started:
            data_started = True
            continue

        if not data_started:
            continue

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


def get_postgres_pod_in_namespace(namespace: str = "rhoai-model-registries") -> Pod:
    """Get the PostgreSQL pod for model catalog database."""
    postgres_pods = list(Pod.get(namespace=namespace, label_selector="app=model-catalog-postgres"))

    if not postgres_pods:
        # Fallback: try finding by name pattern
        all_pods = list(Pod.get(namespace=namespace))
        postgres_pods = [pod for pod in all_pods if "postgres" in pod.name]

    assert postgres_pods, f"No PostgreSQL pod found in namespace {namespace}"
    return postgres_pods[0]


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


def get_models_from_api(
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    page_size: int = 100,
    source_label: str | None = None,
    additional_params: str = "",
) -> dict[str, Any]:
    """
    Helper method to get models from API with optional filtering

    Args:
        model_catalog_rest_url: REST URL for model catalog
        model_registry_rest_headers: Headers for model registry REST API
        page_size: Number of results per page
        source_label: Source label(s) to filter by (must be comma-separated for multiple filters)
        additional_params: Additional query parameters (e.g., "&filterQuery=name='model_name'")

    Returns:
        Dictionary containing the API response
    """
    url = f"{model_catalog_rest_url[0]}models?pageSize={page_size}"

    if source_label:
        url += f"&sourceLabel={source_label}"

    url += additional_params

    return execute_get_command(url=url, headers=model_registry_rest_headers)
