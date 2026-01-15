from typing import Any
import subprocess
import yaml

from kubernetes.dynamic import DynamicClient
from simple_logger.logger import get_logger
from timeout_sampler import retry

from ocp_resources.pod import Pod
from ocp_resources.config_map import ConfigMap
from tests.model_registry.model_catalog.constants import DEFAULT_CATALOGS
from utilities.constants import Timeout

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
    resource = list(kind.get(namespace=namespace, label_selector="component=model-catalog", client=admin_client))
    assert resource
    LOGGER.info(f"Validating resource: {kind}: Found {len(resource)}")
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


# New utility functions for model inclusion/exclusion and cleanup testing


def get_models_from_database_by_source(source_id: str, namespace: str) -> set[str]:
    """
    Query database directly to get all model names for a specific source.

    Args:
        source_id: Catalog source ID to filter by
        namespace: OpenShift namespace for database access

    Returns:
        Set of model names found in database for the source
    """
    from tests.model_registry.model_catalog.utils import execute_database_query, parse_psql_output

    query = f"""
    SELECT DISTINCT c.name as model_name
    FROM "Context" c
    WHERE c.type_id = (SELECT id FROM "Type" WHERE name = 'kf.CatalogModel')
    AND EXISTS (
        SELECT 1
        FROM "ContextProperty" cp
        WHERE cp.context_id = c.id
        AND cp.name = 'source_id'
        AND cp.string_value = '{source_id}'
    )
    ORDER BY model_name;
    """

    result = execute_database_query(query=query, namespace=namespace)
    parsed = parse_psql_output(psql_output=result)
    return set(parsed.get("values", []))


def validate_model_filtering_consistency(
    api_models: set[str], db_models: set[str], source_id: str = "redhat_ai_models"
) -> tuple[bool, str]:
    """
    Validate consistency between API response and database state for model filtering.

    Args:
        api_models: Set of model names from API response
        db_models: Set of model names from database query
        source_id: Source ID for logging context

    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    if api_models != db_models:
        extra_in_api = api_models - db_models
        extra_in_db = db_models - api_models
        return (
            False,
            f"API and DB inconsistency for {source_id}. Extra in API: {extra_in_api}, Extra in DB: {extra_in_db}",
        )

    return True, "Validation passed"


def apply_inclusion_exclusion_filters_to_source(
    admin_client, namespace: str, source_id: str, included_models: list[str] = None, excluded_models: list[str] = None
) -> dict:
    """
    Patch a catalog source with inclusion/exclusion filters.
    First ensures the source exists by syncing from default sources if necessary.

    Args:
        admin_client: OpenShift dynamic client
        namespace: Model registry namespace
        source_id: Source ID to patch
        included_models: List of inclusion patterns
        excluded_models: List of exclusion patterns

    Returns:
        Dictionary with patch information
    """
    from tests.model_registry.constants import DEFAULT_CUSTOM_MODEL_CATALOG

    # Get current ConfigMap (model-catalog-sources)
    sources_cm = ConfigMap(
        name=DEFAULT_CUSTOM_MODEL_CATALOG,
        client=admin_client,
        namespace=namespace,
    )

    # Parse existing sources
    current_yaml = sources_cm.instance.data.get("sources.yaml", "")
    sources_config = yaml.safe_load(current_yaml) if current_yaml else {"catalogs": []}

    # Find the target source
    target_source = None
    for source in sources_config.get("catalogs", []):
        if source.get("id") == source_id:
            target_source = source
            break

    # If source not found, sync from default sources ConfigMap
    if not target_source:
        LOGGER.info(f"Source {source_id} not found in {DEFAULT_CUSTOM_MODEL_CATALOG}. Syncing from default sources.")

        # Get default sources ConfigMap (model-catalog-default-sources)
        default_sources_cm = ConfigMap(
            name="model-catalog-default-sources",
            client=admin_client,
            namespace=namespace,
        )

        # Parse default sources
        default_yaml = default_sources_cm.instance.data.get("sources.yaml", "")
        default_config = yaml.safe_load(default_yaml) if default_yaml else {"catalogs": []}

        # Find source in default sources
        default_target_source = None
        for source in default_config.get("catalogs", []):
            if source.get("id") == source_id:
                default_target_source = source
                break

        if not default_target_source:
            raise ValueError(f"Source {source_id} not found in either ConfigMap")

        # Add all default catalogs to sources_config if not already present
        existing_ids = {source.get("id") for source in sources_config.get("catalogs", [])}
        for default_catalog in default_config.get("catalogs", []):
            if default_catalog.get("id") not in existing_ids:
                sources_config.setdefault("catalogs", []).append(default_catalog)

        # Now find the target source in the updated config
        for source in sources_config.get("catalogs", []):
            if source.get("id") == source_id:
                target_source = source
                break

    # Apply filters
    if included_models is not None:
        if len(included_models) == 0:
            target_source["includedModels"] = []
        else:
            target_source["includedModels"] = included_models
    elif "includedModels" in target_source:
        del target_source["includedModels"]

    if excluded_models is not None:
        if len(excluded_models) == 0:
            target_source["excludedModels"] = []
        else:
            target_source["excludedModels"] = excluded_models
    elif "excludedModels" in target_source:
        del target_source["excludedModels"]

    # Generate new YAML
    new_yaml = yaml.dump(sources_config, default_flow_style=False)

    return {
        "configmap": sources_cm,
        "patch": {
            "metadata": {"name": sources_cm.name, "namespace": sources_cm.namespace},
            "data": {"sources.yaml": new_yaml},
        },
        "original_yaml": current_yaml,
    }


def get_api_models_by_source_label(
    model_catalog_rest_url: list[str], model_registry_rest_headers: dict[str, str], source_label: str
) -> set[str]:
    """Helper to get current model set from API by source label."""
    from tests.model_registry.model_catalog.utils import get_models_from_catalog_api

    response = get_models_from_catalog_api(
        model_catalog_rest_url=model_catalog_rest_url,
        model_registry_rest_headers=model_registry_rest_headers,
        source_label=source_label,
    )
    return {model["name"] for model in response.get("items", [])}


@retry(
    exceptions_dict={ValueError: [], Exception: []},  # Retry on assertion failures and API errors
    wait_timeout=Timeout.TIMEOUT_5MIN,  # 300 seconds default
    sleep=10,  # 10 second poll interval
)
def wait_for_model_count_change(
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    source_label: str,
    expected_count: int,
) -> None:
    """
    Wait for model count to reach expected value using @retry decorator.

    Args:
        model_catalog_rest_url: API URL list
        model_registry_rest_headers: API headers
        source_label: Source to query
        expected_count: Expected number of models

    Raises:
        TimeoutExpiredError: If expected count not reached within timeout
        AssertionError: If count doesn't match (retried automatically)
        Exception: If API errors occur (retried automatically)
    """
    current_models = get_api_models_by_source_label(
        model_catalog_rest_url=model_catalog_rest_url,
        model_registry_rest_headers=model_registry_rest_headers,
        source_label=source_label,
    )
    # Raise AssertionError if condition not met - this will be retried
    if len(current_models) == expected_count:
        return True
    else:
        raise ValueError(f"Expected {expected_count} models, got {len(current_models)}")


@retry(
    exceptions_dict={AssertionError: [], Exception: []},  # Retry on assertion failures and API errors
    wait_timeout=Timeout.TIMEOUT_5MIN,  # 300 seconds default
    sleep=10,  # 10 second poll interval
)
def wait_for_model_set_match(
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    source_label: str,
    expected_models: set[str],
) -> set[str]:
    """
    Wait for specific model set to appear using @retry decorator.

    Args:
        model_catalog_rest_url: API URL list
        model_registry_rest_headers: API headers
        source_label: Source to query
        expected_models: Expected set of model names

    Returns:
        Set of matched models

    Raises:
        TimeoutExpiredError: If expected models not found within timeout
        AssertionError: If models don't match (retried automatically)
        Exception: If API errors occur (retried automatically)
    """
    current_models = get_api_models_by_source_label(
        model_catalog_rest_url=model_catalog_rest_url,
        model_registry_rest_headers=model_registry_rest_headers,
        source_label=source_label,
    )
    # Raise AssertionError if condition not met - this will be retried
    assert current_models == expected_models, f"Expected models {expected_models}, got {current_models}"
    return current_models


def disable_catalog_source(admin_client, namespace: str, source_id: str) -> dict:
    """
    Disable a catalog source by setting enabled: false.
    First ensures the source exists by syncing from default sources if necessary.

    Args:
        admin_client: OpenShift dynamic client
        namespace: Model registry namespace
        source_id: Source ID to disable

    Returns:
        Dictionary with patch information
    """
    from tests.model_registry.constants import DEFAULT_CUSTOM_MODEL_CATALOG

    # Get current ConfigMap (model-catalog-sources)
    sources_cm = ConfigMap(
        name=DEFAULT_CUSTOM_MODEL_CATALOG,
        client=admin_client,
        namespace=namespace,
    )

    # Parse existing sources
    current_yaml = sources_cm.instance.data.get("sources.yaml", "")
    sources_config = yaml.safe_load(current_yaml) if current_yaml else {"catalogs": []}

    # Find the target source
    target_source = None
    for source in sources_config.get("catalogs", []):
        if source.get("id") == source_id:
            target_source = source
            break

    # If source not found, sync from default sources ConfigMap
    if not target_source:
        LOGGER.info(f"Source {source_id} not found in {DEFAULT_CUSTOM_MODEL_CATALOG}. Syncing from default sources.")

        # Get default sources ConfigMap (model-catalog-default-sources)
        default_sources_cm = ConfigMap(
            name="model-catalog-default-sources",
            client=admin_client,
            namespace=namespace,
        )

        # Parse default sources
        default_yaml = default_sources_cm.instance.data.get("sources.yaml", "")
        default_config = yaml.safe_load(default_yaml) if default_yaml else {"catalogs": []}

        # Find source in default sources
        default_target_source = None
        for source in default_config.get("catalogs", []):
            if source.get("id") == source_id:
                default_target_source = source
                break

        if not default_target_source:
            raise ValueError(f"Source {source_id} not found in either ConfigMap")

        # Add all default catalogs to sources_config if not already present
        existing_ids = {source.get("id") for source in sources_config.get("catalogs", [])}
        for default_catalog in default_config.get("catalogs", []):
            if default_catalog.get("id") not in existing_ids:
                sources_config.setdefault("catalogs", []).append(default_catalog)

        # Now find the target source in the updated config
        for source in sources_config.get("catalogs", []):
            if source.get("id") == source_id:
                target_source = source
                break

    # Disable the source
    target_source["enabled"] = False

    # Generate new YAML
    new_yaml = yaml.dump(sources_config, default_flow_style=False)

    return {
        "configmap": sources_cm,
        "patch": {
            "metadata": {"name": sources_cm.name, "namespace": sources_cm.namespace},
            "data": {"sources.yaml": new_yaml},
        },
        "original_yaml": current_yaml,
    }


@retry(
    exceptions_dict={subprocess.CalledProcessError: [], AssertionError: []},
    wait_timeout=Timeout.TIMEOUT_2MIN,  # 120 seconds default
    sleep=5,  # 5 second poll interval
)
def validate_cleanup_logging(
    namespace: str,
    expected_log_patterns: list[str],
) -> list[str]:
    """
    Validate that model cleanup operations are properly logged using @retry decorator.

    Args:
        namespace: Model registry namespace
        expected_log_patterns: List of patterns to find in logs

    Returns:
        List of found patterns

    Raises:
        TimeoutExpiredError: If not all patterns found within timeout
        subprocess.CalledProcessError: If oc command fails (retried automatically)
        AssertionError: If patterns not found (retried automatically)
    """
    import re

    # Get model registry pod logs
    result = subprocess.run(
        args=["oc", "logs", "-n", namespace, "-l", "app.kubernetes.io/name=model-registry", "--tail=100"],
        capture_output=True,
        text=True,
        check=True,
    )

    log_content = result.stdout
    found_patterns = []

    # Check for expected patterns
    for pattern in expected_log_patterns:
        if re.search(pattern, log_content, re.IGNORECASE):
            found_patterns.append(pattern)

    # Raise AssertionError if not all patterns found - this will be retried
    assert len(found_patterns) == len(expected_log_patterns), (
        f"Expected {len(expected_log_patterns)} log patterns, found {len(found_patterns)}: {found_patterns}"
    )

    return found_patterns


def validate_invalid_pattern_error(
    admin_client,
    namespace: str,
    source_id: str,
    invalid_patterns: list[str],
    field_name: str,  # "includedModels" or "excludedModels"
) -> tuple[bool, str]:
    """
    Test that invalid patterns generate appropriate validation errors.

    Args:
        admin_client: OpenShift dynamic client
        namespace: Model registry namespace
        source_id: Source ID to test
        invalid_patterns: List of invalid patterns to test
        field_name: Field name being tested

    Returns:
        Tuple of (error_detected: bool, error_message: str)
    """
    try:
        filters = {field_name: invalid_patterns}
        patch_info = apply_inclusion_exclusion_filters_to_source(
            admin_client=admin_client, namespace=namespace, source_id=source_id, **filters
        )

        # Apply the patch to see if validation catches it
        patch_info["configmap"].update(patch_info["patch"])

        # If we get here without exception, the pattern was accepted
        # This might be expected behavior, so return False for error_detected
        return False, "Pattern was accepted by validation"

    except ValueError as e:
        # Expected validation error
        return True, str(e)
    except Exception as e:
        # Unexpected error type
        return True, f"Unexpected error: {str(e)}"


# Helper functions for model filtering


def filter_models_by_pattern(all_models: set[str], pattern: str) -> set[str]:
    """Helper function to filter models by a given pattern."""
    return {model for model in all_models if pattern in model}
