from typing import Any
import subprocess
import yaml
import re

import pytest
from kubernetes.dynamic import DynamicClient
from simple_logger.logger import get_logger
from timeout_sampler import retry, TimeoutExpiredError

from ocp_resources.pod import Pod
from ocp_resources.config_map import ConfigMap
from ocp_resources.resource import ResourceEditor
from tests.model_registry.model_catalog.constants import (
    DEFAULT_CATALOGS,
    REDHAT_AI_CATALOG_ID,
    REDHAT_AI_CATALOG_NAME,
)
from tests.model_registry.constants import DEFAULT_CUSTOM_MODEL_CATALOG
from tests.model_registry.utils import get_model_catalog_pod, wait_for_model_catalog_api
from utilities.constants import Timeout
from tests.model_registry.model_catalog.utils import (
    get_models_from_catalog_api,
    execute_database_query,
    parse_psql_output,
)
from tests.model_registry.model_catalog.db_constants import GET_MODELS_BY_SOURCE_ID_DB_QUERY

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


def get_models_from_database_by_source(source_id: str, namespace: str) -> set[str]:
    """
    Query database directly to get all model names for a specific source.

    Args:
        source_id: Catalog source ID to filter by
        namespace: OpenShift namespace for database access

    Returns:
        Set of model names found in database for the source
    """

    query = GET_MODELS_BY_SOURCE_ID_DB_QUERY.format(source_id=source_id)
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


def modify_catalog_source(
    admin_client: DynamicClient,
    namespace: str,
    source_id: str,
    enabled: bool = None,
    included_models: list[str] = None,
    excluded_models: list[str] = None,
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

    log_content = model_catalog_pod.log(container="catalog", tail_lines=200)
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


def execute_inclusion_exclusion_filter_test(
    filter_type: str,
    pattern: str,
    filter_value: str,
    baseline_models: set[str],
    admin_client: DynamicClient,
    model_registry_namespace: str,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> None:
    """
    Common implementation for inclusion/exclusion filter tests.

    Args:
        filter_type: "inclusion" or "exclusion"
        pattern: Pattern to match in model names (e.g. "granite", "prometheus")
        filter_value: Filter value to apply (e.g. "*granite*", "*prometheus*")
        baseline_models: Set of all available models from baseline
        admin_client: Kubernetes dynamic client
        model_registry_namespace: Model registry namespace
        model_catalog_rest_url: Model catalog REST API URLs
        model_registry_rest_headers: Headers for model registry requests
    """
    # Calculate expected models based on filter type
    if filter_type == "inclusion":
        expected_models = filter_models_by_pattern(all_models=baseline_models, pattern=pattern)
        test_description = f"{pattern} model inclusion filter"
    else:  # exclusion
        pattern_models = filter_models_by_pattern(all_models=baseline_models, pattern=pattern)
        expected_models = baseline_models - pattern_models
        test_description = f"{pattern} model exclusion filter"

    LOGGER.info(f"Testing {test_description}")

    # Apply filter based on type
    if filter_type == "inclusion":
        patch_info = modify_catalog_source(
            admin_client=admin_client,
            namespace=model_registry_namespace,
            source_id=REDHAT_AI_CATALOG_ID,
            included_models=[filter_value],
        )
    else:  # exclusion
        patch_info = modify_catalog_source(
            admin_client=admin_client,
            namespace=model_registry_namespace,
            source_id=REDHAT_AI_CATALOG_ID,
            excluded_models=[filter_value],
        )

    with ResourceEditor(patches={patch_info["configmap"]: patch_info["patch"]}):
        wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

        try:
            api_models = wait_for_model_set_match(
                model_catalog_rest_url=model_catalog_rest_url,
                model_registry_rest_headers=model_registry_rest_headers,
                source_label=REDHAT_AI_CATALOG_NAME,
                expected_models=expected_models,
            )
        except TimeoutExpiredError as e:
            pytest.fail(f"Timeout waiting for {pattern} models to appear. Expected: {expected_models}, {e}")

        db_models = get_models_from_database_by_source(
            source_id=REDHAT_AI_CATALOG_ID, namespace=model_registry_namespace
        )

        # Validate consistency
        is_valid, error_msg = validate_model_filtering_consistency(api_models=api_models, db_models=db_models)
        assert is_valid, error_msg

        # Validate expected models match actual
        assert api_models == expected_models, f"Expected {test_description}: {expected_models}, got {api_models}"

        LOGGER.info(f"SUCCESS: {len(api_models)} {pattern} models {filter_type}")


@retry(wait_timeout=300, sleep=10, exceptions_dict={Exception: []}, print_log=False)
def _validate_baseline_models(
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    model_registry_namespace: str,
    expected_models: set[str],
    expected_count: int,
) -> None:
    """
    Validate that baseline model expectations are met.
    Raises exception if validation fails (triggers retry).
    Returns None if successful (stops retry).
    """
    # Fetch current models from API
    api_response = get_models_from_catalog_api(
        model_catalog_rest_url=model_catalog_rest_url,
        model_registry_rest_headers=model_registry_rest_headers,
        source_label="Red Hat AI",
    )
    api_models = {model["name"] for model in api_response.get("items", [])}

    # Fetch current models from database
    db_models = get_models_from_database_by_source(source_id=REDHAT_AI_CATALOG_ID, namespace=model_registry_namespace)

    count = len(api_models)

    # Validate all expectations - raise on any failure
    if count != expected_count:
        raise AssertionError(f"Expected {expected_count} models, got {count}")

    if api_models != db_models:
        raise AssertionError(f"API models {api_models} don't match database models {db_models}")

    if api_models != expected_models:
        raise AssertionError(f"Models {api_models} don't match expected set {expected_models}")

    # Additional category validation
    granite_models = {model for model in api_models if "granite" in model}
    prometheus_models = {model for model in api_models if "prometheus" in model}

    if len(granite_models) != 6 or len(prometheus_models) != 1:
        raise AssertionError(
            f"""Expected 6 granite + 1 prometheus models, \
            got {len(granite_models)} granite + {len(prometheus_models)} prometheus"""
        )

    LOGGER.info("Baseline model validation successful: 7 models (6 granite, 1 prometheus)")
    return True


def ensure_baseline_model_state(
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    model_registry_namespace: str,
) -> None:
    """
    Utility function to ensure that our baseline assumptions about the model data are correct.
    This should be called at the end of tests to ensure state consistency for subsequent tests.
    Uses @retry decorator for automatic polling (300s timeout, 10s interval) and eventual reconciliation.

    Args:
        model_catalog_rest_url: URL for model catalog API
        model_registry_rest_headers: Headers for API requests
        model_registry_namespace: Namespace for model registry

    Raises:
        pytest.FailError: If baseline state cannot be achieved after timeout
    """
    # Expected baseline data
    expected_models = {
        "granite-3.1-8b-lab-v1",
        "granite-7b-redhat-lab",
        "granite-8b-code-base",
        "granite-8b-code-instruct",
        "granite-8b-lab-v1",
        "granite-8b-starter-v1",
        "prometheus-8x7b-v2-0",
    }
    expected_count = 7

    # Use retry decorator for automatic polling and eventual reconciliation
    try:
        _validate_baseline_models(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            model_registry_namespace=model_registry_namespace,
            expected_models=expected_models,
            expected_count=expected_count,
        )
    except TimeoutExpiredError:
        pytest.fail("Failed to restore baseline model state after 300s timeout")
