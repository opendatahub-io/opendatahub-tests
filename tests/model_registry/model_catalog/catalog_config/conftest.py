import pytest
from typing import Generator

from kubernetes.dynamic import DynamicClient
from ocp_resources.resource import ResourceEditor
from tests.model_registry.model_catalog.constants import REDHAT_AI_CATALOG_ID, REDHAT_AI_CATALOG_NAME
from tests.model_registry.model_catalog.catalog_config.utils import (
    filter_models_by_pattern,
    modify_catalog_source,
    wait_for_catalog_source_restore,
)
from tests.model_registry.utils import wait_for_model_catalog_api, wait_for_model_catalog_pod_ready_after_deletion


@pytest.fixture(scope="function")
def redhat_ai_models_with_inclusion_filter(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    baseline_redhat_ai_models: dict[str, set[str] | int],
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    catalog_pod_model_counts: dict[str, int],
) -> Generator[set[str], None, None]:
    """
    Fixture that applies inclusion filter to redhat_ai catalog and yields expected models.

    Expects request.param dict with:
    - "pattern": Pattern to match in model names
    - "filter_value": Filter value for inclusion (e.g., "*granite*")

    Returns:
        set[str]: Expected redhat_ai models after applying the inclusion filter
    """
    # Get parameters from test
    param = getattr(request, "param", {})
    baseline_models = baseline_redhat_ai_models["api_models"]

    # Modify catalog source with inclusion filter
    patch_info = modify_catalog_source(
        admin_client=admin_client,
        namespace=model_registry_namespace,
        source_id=REDHAT_AI_CATALOG_ID,
        included_models=[param.get("filter_value")],
    )

    with ResourceEditor(patches={patch_info["configmap"]: patch_info["patch"]}):
        wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

        yield filter_models_by_pattern(all_models=baseline_models, pattern=param.get("pattern"))

    # Ensure baseline model state is restored for subsequent tests
    wait_for_catalog_source_restore(
        model_catalog_rest_url=model_catalog_rest_url,
        model_registry_rest_headers=model_registry_rest_headers,
        source_label=REDHAT_AI_CATALOG_NAME,
        expected_count=catalog_pod_model_counts[REDHAT_AI_CATALOG_ID],
    )


@pytest.fixture(scope="function")
def redhat_ai_models_with_exclusion_filter(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    baseline_redhat_ai_models: dict[str, set[str] | int],
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    catalog_pod_model_counts: dict[str, int],
) -> Generator[set[str], None, None]:
    """
    Fixture that applies exclusion filter to redhat_ai catalog and yields expected models.

    Expects request.param dict with:
    - "pattern": Pattern to match in model names for exclusion
    - "filter_value": Filter value for exclusion (e.g., "*granite*")
    - "log_cleanup": Optional boolean (default False) - if True, adds pod readiness checks for cleanup logging

    Returns:
        set[str]: Expected redhat_ai models after applying the exclusion filter
    """
    # Get parameters from test
    param = getattr(request, "param", {})
    baseline_models = baseline_redhat_ai_models["api_models"]
    log_cleanup = param.get("log_cleanup", False)

    # Calculate expected models by excluding those that match the pattern
    models_to_exclude = filter_models_by_pattern(all_models=baseline_models, pattern=param.get("pattern"))
    expected_models = baseline_models - models_to_exclude

    # Modify catalog source with exclusion filter
    patch_info = modify_catalog_source(
        admin_client=admin_client,
        namespace=model_registry_namespace,
        source_id=REDHAT_AI_CATALOG_ID,
        excluded_models=[param.get("filter_value")],
    )

    with ResourceEditor(patches={patch_info["configmap"]: patch_info["patch"]}):
        wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

        # Add pod readiness checks if log_cleanup is requested
        if log_cleanup:
            wait_for_model_catalog_pod_ready_after_deletion(
                client=admin_client, model_registry_namespace=model_registry_namespace
            )
            wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

        yield expected_models

    # Ensure baseline model state is restored for subsequent tests
    wait_for_catalog_source_restore(
        model_catalog_rest_url=model_catalog_rest_url,
        model_registry_rest_headers=model_registry_rest_headers,
        source_label=REDHAT_AI_CATALOG_NAME,
        expected_count=catalog_pod_model_counts[REDHAT_AI_CATALOG_ID],
    )


@pytest.fixture(scope="function")
def redhat_ai_models_with_combined_filter(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    baseline_redhat_ai_models: dict[str, set[str] | int],
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    catalog_pod_model_counts: dict[str, int],
) -> Generator[set[str], None, None]:
    """
    Fixture that applies combined inclusion and exclusion filters to redhat_ai catalog.

    Expects request.param dict with:
    - "include_pattern": Pattern to match for inclusion
    - "include_filter_value": Filter value for inclusion (e.g., "*granite*")
    - "exclude_pattern": Pattern to match for exclusion from included set
    - "exclude_filter_value": Filter value for exclusion (e.g., "*lab*")

    Returns:
        set[str]: Expected redhat_ai models after applying both filters
    """
    # Get parameters from test
    param = getattr(request, "param", {})
    baseline_models = baseline_redhat_ai_models["api_models"]

    include_pattern = param.get("include_pattern")
    exclude_pattern = param.get("exclude_pattern")
    include_filter_value = param.get("include_filter_value")
    exclude_filter_value = param.get("exclude_filter_value")

    # Calculate expected models: first include, then exclude from included set
    included_models = filter_models_by_pattern(all_models=baseline_models, pattern=include_pattern)
    expected_models = {model for model in included_models if exclude_pattern not in model}

    # Modify catalog source with both inclusion and exclusion filters
    patch_info = modify_catalog_source(
        admin_client=admin_client,
        namespace=model_registry_namespace,
        source_id=REDHAT_AI_CATALOG_ID,
        included_models=[include_filter_value],
        excluded_models=[exclude_filter_value],
    )

    with ResourceEditor(patches={patch_info["configmap"]: patch_info["patch"]}):
        wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

        yield expected_models

    # Ensure baseline model state is restored for subsequent tests
    wait_for_catalog_source_restore(
        model_catalog_rest_url=model_catalog_rest_url,
        model_registry_rest_headers=model_registry_rest_headers,
        source_label=REDHAT_AI_CATALOG_NAME,
        expected_count=catalog_pod_model_counts[REDHAT_AI_CATALOG_ID],
    )


@pytest.fixture(scope="class")
def disabled_redhat_ai_source(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    catalog_pod_model_counts: dict[str, int],
) -> Generator[None, None, None]:
    """
    Fixture that disables the redhat_ai catalog source and yields control.

    Automatically restores the source to enabled state after test completion.
    """
    # Disable the source
    disable_patch = modify_catalog_source(
        admin_client=admin_client,
        namespace=model_registry_namespace,
        source_id=REDHAT_AI_CATALOG_ID,
        enabled=False,
    )

    with ResourceEditor(patches={disable_patch["configmap"]: disable_patch["patch"]}):
        wait_for_model_catalog_pod_ready_after_deletion(
            client=admin_client, model_registry_namespace=model_registry_namespace
        )
        wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

        yield
    wait_for_catalog_source_restore(
        model_catalog_rest_url=model_catalog_rest_url,
        model_registry_rest_headers=model_registry_rest_headers,
        source_label=REDHAT_AI_CATALOG_NAME,
        expected_count=catalog_pod_model_counts[REDHAT_AI_CATALOG_ID],
    )
