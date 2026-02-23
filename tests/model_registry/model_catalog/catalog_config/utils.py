import re
import subprocess
from typing import Any

import pytest
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutExpiredError, retry

from tests.model_registry.constants import DEFAULT_CUSTOM_MODEL_CATALOG
from tests.model_registry.model_catalog.constants import (
    DEFAULT_CATALOGS,
    REDHAT_AI_CATALOG_ID,
    REDHAT_AI_CATALOG_NAME,
)
from tests.model_registry.model_catalog.db_constants import GET_MODELS_BY_SOURCE_ID_DB_QUERY
from tests.model_registry.model_catalog.utils import (
    execute_database_query,
    get_models_from_catalog_api,
    parse_psql_output,
)
from tests.model_registry.utils import get_model_catalog_pod
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


def get_models_from_database_by_source(admin_client: DynamicClient, source_id: str, namespace: str) -> set[str]:
    """
    Query database directly to get all model names for a specific source.

    Args:
        source_id: Catalog source ID to filter by
        namespace: OpenShift namespace for database access

    Returns:
        Set of model names found in database for the source
    """

    query = GET_MODELS_BY_SOURCE_ID_DB_QUERY.format(source_id=source_id)
    result = execute_database_query(admin_client=admin_client, query=query, namespace=namespace)
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


def validate_filter_test_result(
    admin_client: DynamicClient,
    expected_models: set[str],
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    model_registry_namespace: str,
    filter_type: str = "filter",
) -> None:
    """
    Utility function to validate filtering test results.

    Performs common validation steps:
    1. Wait for API models to match expected set
    2. Get database models
    3. Validate API/DB consistency
    4. Assert expected models match actual
    5. Log success message

    Args:
        admin_client: DynamicClient to connect to OpenShift
        expected_models: Set of expected model names after filtering
        model_catalog_rest_url: Model catalog REST API URL
        model_registry_rest_headers: Headers for API requests
        model_registry_namespace: Kubernetes namespace
        filter_type: Type of filter for logging (e.g., "inclusion", "exclusion")
    """
    # Wait for API models to match expected set
    api_models = wait_for_model_set_match(
        model_catalog_rest_url=model_catalog_rest_url,
        model_registry_rest_headers=model_registry_rest_headers,
        source_label=REDHAT_AI_CATALOG_NAME,
        expected_models=expected_models,
    )

    # Get database models
    db_models = get_models_from_database_by_source(
        admin_client=admin_client, source_id=REDHAT_AI_CATALOG_ID, namespace=model_registry_namespace
    )

    # Validate consistency between API and database
    is_valid, error_msg = validate_model_filtering_consistency(api_models=api_models, db_models=db_models)
    assert is_valid, error_msg

    # Validate expected models match actual
    assert api_models == expected_models, f"Expected models: {expected_models}, got {api_models}"

    LOGGER.info(f"SUCCESS: {len(api_models)} models after {filter_type} filter")


def validate_source_disabling_result(
    admin_client: DynamicClient,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    model_registry_namespace: str,
) -> None:
    """
    Utility function to validate source disabling test results.

    Performs validation steps:
    1. Wait for all models to be removed (count = 0)
    2. Verify database is cleaned
    3. Log success message

    Args:
        model_catalog_rest_url: Model catalog REST API URL
        model_registry_rest_headers: Headers for API requests
        model_registry_namespace: Kubernetes namespace
    """
    # Wait for models to be removed
    try:
        wait_for_model_count_change(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_label=REDHAT_AI_CATALOG_NAME,
            expected_count=0,
        )
    except TimeoutExpiredError as e:
        pytest.fail(f"Expected all models to be removed when source is disabled: {e}")

    # Verify database is also cleaned
    db_models = get_models_from_database_by_source(
        admin_client=admin_client, source_id=REDHAT_AI_CATALOG_ID, namespace=model_registry_namespace
    )
    assert len(db_models) == 0, f"Database should be clean when source disabled, found: {db_models}"

    LOGGER.info("SUCCESS: Source disabling removed all models")


def modify_catalog_source(
    admin_client: DynamicClient,
    namespace: str,
    source_id: str,
    enabled: bool | None = None,
    included_models: list[str] | None = None,
    excluded_models: list[str] | None = None,
) -> dict[str, ConfigMap | dict[str, Any] | str]:
    """
    Modify a catalog source with various configuration changes.
    First ensures the source exists by syncing from default sources if necessary.

    Args:
        admin_client: OpenShift dynamic client
        namespace: Model registry namespace
        source_id: Source ID to modify
        enabled: Set to False to disable the source, True to enable, None to leave unchanged
        included_models: List of inclusion patterns (None = no change, [] = clear)
        excluded_models: List of exclusion patterns (None = no change, [] = clear)

    Returns:
        Dictionary with patch information
    """
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

    # Apply modifications
    if enabled is not None:
        target_source["enabled"] = enabled

    if included_models is not None:
        if len(included_models) == 0:
            target_source["includedModels"] = []
        else:
            target_source["includedModels"] = included_models

    if excluded_models is not None:
        if len(excluded_models) == 0:
            target_source["excludedModels"] = []
        else:
            target_source["excludedModels"] = excluded_models

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
    response = get_models_from_catalog_api(
        model_catalog_rest_url=model_catalog_rest_url,
        model_registry_rest_headers=model_registry_rest_headers,
        source_label=source_label,
    )
    return {model["name"] for model in response.get("items", [])}


@retry(
    exceptions_dict={ValueError: [], Exception: []},
    wait_timeout=Timeout.TIMEOUT_5MIN,
    sleep=10,
)
def wait_for_model_count_change(
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    source_label: str,
    expected_count: int,
) -> bool:
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
    if len(current_models) == expected_count:
        return True
    else:
        raise ValueError(f"Expected {expected_count} models, got {len(current_models)}")


@retry(
    exceptions_dict={AssertionError: [], Exception: []},
    wait_timeout=Timeout.TIMEOUT_5MIN,
    sleep=10,
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


@retry(
    exceptions_dict={subprocess.CalledProcessError: [], AssertionError: []},
    wait_timeout=Timeout.TIMEOUT_2MIN,
    sleep=5,
)
def validate_cleanup_logging(
    client: DynamicClient,
    namespace: str,
    expected_log_patterns: list[str],
) -> list[re.Match[str]]:
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
    model_catalog_pod = get_model_catalog_pod(
        client=client, model_registry_namespace=namespace, label_selector="app=model-catalog"
    )[0]

    log_content = model_catalog_pod.log(container="catalog")
    found_patterns = []

    # Check for expected patterns
    for pattern in expected_log_patterns:
        found = re.search(pattern, log_content, re.IGNORECASE)
        if found:
            found_patterns.append(found)

    return found_patterns


def filter_models_by_pattern(all_models: set[str], pattern: str) -> set[str]:
    """Helper function to filter models by a given pattern."""
    return {model for model in all_models if pattern in model}


@retry(wait_timeout=300, sleep=10, exceptions_dict={Exception: []}, print_log=False)
def wait_for_catalog_source_restore(
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    expected_count: int,
    source_label: str,
) -> bool:
    """
    Waits for the source api to return a specified number of models as expected
    """
    # Fetch current models from API
    api_response = get_models_from_catalog_api(
        model_catalog_rest_url=model_catalog_rest_url,
        model_registry_rest_headers=model_registry_rest_headers,
        source_label="Red Hat AI",
        page_size=1000,
    )
    model_count = api_response.get("size")
    LOGGER.warning(f"Model count: {model_count}, expected {expected_count}")
    # Validate all expectations - raise on any failure
    if model_count != expected_count:
        raise AssertionError(f"Expected {expected_count} models, got {model_count}")

    LOGGER.info("Found expected number of models: %s for source: %s", expected_count, source_label)
    return True
