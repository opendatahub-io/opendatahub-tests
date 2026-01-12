from typing import Any, Tuple, List, Dict
import json
import yaml
import requests
from fnmatch import fnmatch

from kubernetes.dynamic import DynamicClient
from simple_logger.logger import get_logger

from ocp_resources.pod import Pod
from ocp_resources.config_map import ConfigMap
from tests.model_registry.model_catalog.constants import DEFAULT_CATALOGS, HF_MODELS, CATALOG_CONTAINER
from tests.model_registry.constants import DEFAULT_CUSTOM_MODEL_CATALOG, DEFAULT_MODEL_CATALOG_CM
from tests.model_registry.utils import execute_get_command, get_rest_headers

LOGGER = get_logger(name=__name__)


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


def get_labels_from_configmaps(admin_client: DynamicClient, namespace: str) -> List[Dict[str, Any]]:
    """
    Get all labels from both model catalog ConfigMaps.

    Args:
        admin_client: Kubernetes client
        namespace: Namespace containing the ConfigMaps

    Returns:
        List of all label dictionaries from both ConfigMaps
    """
    labels = []

    # Get labels from default ConfigMap
    default_cm = ConfigMap(name=DEFAULT_MODEL_CATALOG_CM, client=admin_client, namespace=namespace)
    default_data = yaml.safe_load(default_cm.instance.data["sources.yaml"])
    if "labels" in default_data:
        labels.extend(default_data["labels"])

    # Get labels from sources ConfigMap
    sources_cm = ConfigMap(name=DEFAULT_CUSTOM_MODEL_CATALOG, client=admin_client, namespace=namespace)
    sources_data = yaml.safe_load(sources_cm.instance.data["sources.yaml"])
    if "labels" in sources_data:
        labels.extend(sources_data["labels"])

    return labels


def get_labels_from_api(model_catalog_rest_url: str, user_token: str) -> List[Dict[str, Any]]:
    """
    Get labels from the API endpoint.

    Args:
        model_catalog_rest_url: Base URL for model catalog API
        user_token: Authentication token

    Returns:
        List of label dictionaries from API response
    """
    url = f"{model_catalog_rest_url}labels"
    headers = get_rest_headers(token=user_token)
    response = execute_get_command(url=url, headers=headers)
    return response["items"]


def verify_labels_match(expected_labels: List[Dict[str, Any]], api_labels: List[Dict[str, Any]]) -> None:
    """
    Verify that all expected labels are present in the API response.

    Args:
        expected_labels: Labels expected from ConfigMaps
        api_labels: Labels returned by API

    Raises:
        AssertionError: If any expected label is not found in API response
    """
    LOGGER.info(f"Verifying {len(expected_labels)} expected labels against {len(api_labels)} API labels")

    for expected_label in expected_labels:
        found = False
        for api_label in api_labels:
            if (
                expected_label.get("name") == api_label.get("name")
                and expected_label.get("displayName") == api_label.get("displayName")
                and expected_label.get("description") == api_label.get("description")
            ):
                found = True
                break

        assert found, f"Expected label not found in API response: {expected_label}"


def get_hf_catalog_str(ids: list[str]) -> str:
    """
    Generate a HuggingFace catalog configuration string in YAML format.
    Similar to get_catalog_str() but for HuggingFace catalogs.

    Args:
        ids (list): List of model set identifiers that correspond to keys in MODELS dict

    Returns:
        str: YAML formatted catalog configuration with multiple catalog entries
    """
    catalog_entries = ""

    for source_id in ids:
        if source_id not in HF_MODELS:
            raise ValueError(f"Model ID '{source_id}' not found in MODELS dictionary")
        name = f"HuggingFace Source {source_id}"
        # Build catalog entry
        catalog_entries += f"""
- name: {name}
  id: huggingface_{source_id}
  type: "hf"
  enabled: true
  includedModels:
  {get_included_model_str(models=HF_MODELS[source_id])}
  labels:
  - {name}
"""

    # Combine all catalog entries
    return f"""catalogs:
    {catalog_entries}
    """


def get_included_model_str(models: list[str]) -> str:
    included_models: str = ""
    for model_name in models:
        included_models += f"""
    - {model_name}
"""
    return included_models


def extract_custom_property_values(custom_properties: dict[str, Any]) -> dict[str, str]:
    """
    Extract string values from MetadataStringValue format for custom properties.

    Args:
        custom_properties: Dictionary of custom properties from API response

    Returns:
        Dictionary of extracted string values for size, tensor_type, variant_group_id
    """
    extracted = {}
    expected_keys = ["size", "tensor_type", "variant_group_id"]

    for key in expected_keys:
        if key in custom_properties:
            prop_data = custom_properties[key]
            if isinstance(prop_data, dict) and "string_value" in prop_data:
                extracted[key] = prop_data["string_value"]
            else:
                LOGGER.warning(f"Unexpected format for custom property '{key}': {prop_data}")

    LOGGER.info(f"Extracted {len(extracted)} custom properties: {list(extracted.keys())}")
    return extracted


def validate_custom_properties_structure(custom_properties: dict[str, Any]) -> bool:
    """
    Validate that custom properties follow the expected MetadataStringValue structure.

    Args:
        custom_properties: Dictionary of custom properties from API response

    Returns:
        True if all custom properties have valid structure, False otherwise
    """
    if not custom_properties:
        LOGGER.info("No custom properties found - structure validation skipped")
        return True

    expected_keys = ["size", "tensor_type", "variant_group_id"]

    for key in expected_keys:
        if key in custom_properties:
            prop_data = custom_properties[key]

            if not isinstance(prop_data, dict):
                LOGGER.error(f"Custom property '{key}' is not a dictionary: {prop_data}")
                return False

            if "metadataType" not in prop_data:
                LOGGER.error(f"Custom property '{key}' missing 'metadataType' field")
                return False

            if prop_data.get("metadataType") != "MetadataStringValue":
                LOGGER.error(f"Custom property '{key}' has unexpected metadataType: {prop_data.get('metadataType')}")
                return False

            if "string_value" not in prop_data:
                LOGGER.error(f"Custom property '{key}' missing 'string_value' field")
                return False

            if not isinstance(prop_data.get("string_value"), str):
                LOGGER.error(f"Custom property '{key}' string_value is not a string: {prop_data.get('string_value')}")
                return False

            LOGGER.info(f"Custom property '{key}' has valid structure: '{prop_data.get('string_value')}'")

    LOGGER.info("All custom properties have valid structure")
    return True


def validate_custom_properties_match_metadata(api_custom_properties: dict[str, str], metadata: dict[str, Any]) -> bool:
    """
    Compare API custom properties with metadata.json values.

    Args:
        api_custom_properties: Extracted custom properties from API (string values)
        metadata: Parsed metadata.json content

    Returns:
        True if all custom properties match metadata values, False otherwise
    """
    expected_keys = ["size", "tensor_type", "variant_group_id"]

    for key in expected_keys:
        api_value = api_custom_properties.get(key)
        metadata_value = metadata.get(key)

        if api_value != metadata_value:
            LOGGER.error(f"Mismatch for custom property '{key}': API='{api_value}' vs metadata='{metadata_value}'")
            return False

        if api_value is not None:  # Only log if the property exists
            LOGGER.info(f"Custom property '{key}' matches: '{api_value}'")

    LOGGER.info("All custom properties match metadata.json values")
    return True


def get_metadata_from_catalog_pod(model_catalog_pod: Pod, model_name: str) -> dict[str, Any]:
    """
    Read and parse metadata.json for a model from the catalog pod.

    Args:
        model_catalog_pod: The catalog pod instance
        model_name: Name of the model

    Returns:
        Parsed metadata.json content

    Raises:
        Exception: If metadata.json cannot be read or parsed
    """
    metadata_path = f"/shared-benchmark-data/{model_name}/metadata.json"
    LOGGER.info(f"Reading metadata from: {metadata_path}")

    try:
        metadata_json = model_catalog_pod.execute(command=["cat", metadata_path], container=CATALOG_CONTAINER)
        metadata = json.loads(metadata_json)
        LOGGER.info(f"Successfully loaded metadata.json for model '{model_name}'")
        return metadata
    except Exception as e:
        LOGGER.error(f"Failed to read metadata.json for model '{model_name}': {e}")
        raise


def execute_model_catalog_post_command(url: str, token: str, files: dict[str, tuple[str, str, str]]) -> dict[str, Any]:
    """
    Execute model catalog POST endpoint with multipart/form-data files.

    Args:
        url: API endpoint URL
        token: Authorization bearer token
        files: Dictionary mapping form field names to (filename, content, mime_type) tuples

    Returns:
        dict: Parsed JSON response

    Raises:
        HTTPError: If response status is not successful
    """
    headers = {"Authorization": f"Bearer {token}"}

    LOGGER.info(f"Executing model catalog POST: {url}")
    response = requests.post(url=url, headers=headers, files=files, verify=False, timeout=60)
    response.raise_for_status()
    return response.json()


def build_catalog_preview_config(
    yaml_catalog_path: str | None = None,
    included_patterns: list[str] | None = None,
    excluded_patterns: list[str] | None = None,
) -> str:
    """
    Build catalog preview config YAML content.

    Args:
        yaml_catalog_path: Path to YAML catalog file on the pod (None when using catalogData parameter)
        included_patterns: List of glob patterns for includedModels (None means no filter)
        excluded_patterns: List of glob patterns for excludedModels (None means no filter)

    Returns:
        str: YAML config content for preview API
    """
    config_lines = ["type: yaml"]

    # Only add yamlCatalogPath if provided (not needed when using catalogData)
    if yaml_catalog_path:
        config_lines.extend([
            "properties:",
            f"  yamlCatalogPath: {yaml_catalog_path}",
        ])

    if included_patterns:
        config_lines.append("includedModels:")
        config_lines.extend(f'  - "{pattern}"' for pattern in included_patterns)

    if excluded_patterns:
        config_lines.append("excludedModels:")
        config_lines.extend(f'  - "{pattern}"' for pattern in excluded_patterns)

    return "\n".join(config_lines)


def validate_catalog_preview_counts(
    api_counts: dict[str, int],
    yaml_models: list[dict[str, Any]],
    included_patterns: list[str] | None = None,
    excluded_patterns: list[str] | None = None,
) -> None:
    """
    Validate catalog preview API counts against expected YAML content.

    Args:
        api_counts: Dictionary with 'excludedModels', 'includedModels', 'totalModels'
        yaml_models: List of models from YAML catalog
        included_patterns: List of glob patterns for includedModels (None means include all)
        excluded_patterns: List of glob patterns for excludedModels (None means exclude none)

    Raises:
        AssertionError: If validation fails
    """
    # Apply the same filters to YAML models and get expected counts
    LOGGER.info(f"Found {len(yaml_models)} total models in YAML file")
    expected_counts = filter_models_by_patterns(
        models=yaml_models, included_patterns=included_patterns, excluded_patterns=excluded_patterns
    )

    # Validate API counts match expected counts from YAML - collect all errors
    errors = []

    if api_counts["totalModels"] != expected_counts["totalModels"]:
        errors.append(f"Total mismatch: API={api_counts['totalModels']}, expected={expected_counts['totalModels']}")

    if api_counts["includedModels"] != expected_counts["includedModels"]:
        errors.append(
            f"Included mismatch: API={api_counts['includedModels']}, expected={expected_counts['includedModels']}"
        )

    if api_counts["excludedModels"] != expected_counts["excludedModels"]:
        errors.append(
            f"Excluded mismatch: API={api_counts['excludedModels']}, expected={expected_counts['excludedModels']}"
        )

    assert not errors, "Validation failures:\n" + "\n".join(f"  - {err}" for err in errors)

    LOGGER.info(f"Preview validation passed - API counts match YAML content: {expected_counts}")


def validate_catalog_preview_items(
    result: dict[str, Any],
    included_patterns: list[str] | None = None,
    excluded_patterns: list[str] | None = None,
) -> None:
    """
    Validate that each item in the preview response has the correct 'included' property.

    Args:
        result: API response from preview endpoint
        included_patterns: List of glob patterns for includedModels (None means include all)
        excluded_patterns: List of glob patterns for excludedModels (None means exclude none)

    Raises:
        AssertionError: If any item has incorrect 'included' value
    """
    items = result.get("items", [])
    LOGGER.info(f"Validating 'included' property for {len(items)} items")

    errors = []
    for item in items:
        model_name = item.get("name", "")
        item_included = item.get("included")

        if item_included is None:
            errors.append(f"Model '{model_name}': missing 'included' property")
            continue

        # Use shared logic to determine if model should be included
        expected_included = _should_include_model(
            model_name=model_name, included_patterns=included_patterns, excluded_patterns=excluded_patterns
        )

        if item_included != expected_included:
            errors.append(f"Model '{model_name}': included={item_included}, expected={expected_included}")

    assert not errors, f"Found {len(errors)} items with incorrect 'included' property:\n" + "\n".join(errors)
    LOGGER.info(f"All {len(items)} items have correct 'included' property")


def _should_include_model(
    model_name: str, included_patterns: list[str] | None = None, excluded_patterns: list[str] | None = None
) -> bool:
    """
    Determine if a model should be included based on include/exclude patterns.

    Args:
        model_name: Name of the model to check
        included_patterns: List of glob patterns for includedModels (None means include all)
        excluded_patterns: List of glob patterns for excludedModels (None means exclude none)

    Returns:
        bool: True if model should be included
    """
    # Check if model matches any included pattern
    matches_included = any(fnmatch(model_name, pattern) for pattern in included_patterns) if included_patterns else True

    # Check if model matches any excluded pattern
    matches_excluded = (
        any(fnmatch(model_name, pattern) for pattern in excluded_patterns) if excluded_patterns else False
    )

    # Model is included if it matches include pattern AND does not match exclude pattern
    return matches_included and not matches_excluded


def filter_models_by_patterns(
    models: list[dict[str, Any]], included_patterns: list[str] | None = None, excluded_patterns: list[str] | None = None
) -> dict[str, int]:
    """
    Filter models based on includedModels and excludedModels glob-like patterns.

    Args:
        models: List of model dictionaries with 'name' field
        included_patterns: List of glob patterns for includedModels (None means include all)
        excluded_patterns: List of glob patterns for excludedModels (None means exclude none)

    Returns:
        dict: Dictionary with keys 'includedModels', 'excludedModels', 'totalModels'
    """
    total_models = len(models)
    included_count = 0

    for model in models:
        model_name = model.get("name", "")
        if _should_include_model(
            model_name=model_name, included_patterns=included_patterns, excluded_patterns=excluded_patterns
        ):
            included_count += 1

    excluded_count = total_models - included_count

    LOGGER.info(
        f"Filtered {total_models} models: {included_count} included, {excluded_count} excluded "
        f"(patterns: include={included_patterns}, exclude={excluded_patterns})"
    )

    return {"includedModels": included_count, "excludedModels": excluded_count, "totalModels": total_models}
