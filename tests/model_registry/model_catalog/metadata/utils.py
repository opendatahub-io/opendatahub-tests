import json
from fnmatch import fnmatch
from typing import Any

import requests
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger

from tests.model_registry.constants import DEFAULT_CUSTOM_MODEL_CATALOG, DEFAULT_MODEL_CATALOG_CM
from tests.model_registry.utils import execute_get_command, get_rest_headers

LOGGER = get_logger(name=__name__)
CATALOG_CONTAINER = "catalog"


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
        return metadata  # noqa: TRY300
    except Exception as e:
        LOGGER.error(f"Failed to read metadata.json for model '{model_name}': {e}")
        raise


def compare_filter_options_with_database(
    api_filters: dict[str, Any], db_properties: dict[str, list[str]], excluded_fields: set[str]
) -> tuple[bool, list[str]]:
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
        f"After applying API filtering, expecting {len(expected_properties)}"
        f" properties: {list(expected_properties.keys())}"
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


def get_labels_from_configmaps(admin_client: DynamicClient, namespace: str) -> list[dict[str, Any]]:
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


def get_labels_from_api(model_catalog_rest_url: str, user_token: str) -> list[dict[str, Any]]:
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


def verify_labels_match(expected_labels: list[dict[str, Any]], api_labels: list[dict[str, Any]]) -> None:
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
