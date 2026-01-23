import pytest
from typing import Generator
from simple_logger.logger import get_logger

from kubernetes.dynamic import DynamicClient
from ocp_resources.resource import ResourceEditor
from tests.model_registry.model_catalog.constants import REDHAT_AI_CATALOG_ID, REDHAT_AI_CATALOG_NAME
from tests.model_registry.model_catalog.catalog_config.utils import (
    filter_models_by_pattern,
    modify_catalog_source,
    wait_for_catalog_source_restore,
)
from tests.model_registry.utils import wait_for_model_catalog_api, wait_for_model_catalog_pod_ready_after_deletion

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="function")
def redhat_ai_models_with_filter(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    baseline_redhat_ai_models: dict[str, set[str] | int],
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    catalog_pod_model_counts: dict[str, int],
) -> Generator[set[str], None, None]:
    """
    Unified fixture for applying filters to redhat_ai catalog and yielding expected models.

    Expects request.param dict with:
    - "filter_type": "inclusion", "exclusion", or "combined"
    - For inclusion: "pattern", "filter_value"
    - For exclusion: "pattern", "filter_value", optional "log_cleanup"
    - For combined: "include_pattern", "include_filter_value", "exclude_pattern", "exclude_filter_value"

    Returns:
        set[str]: Expected redhat_ai models after applying the filter(s)
    """
    param = getattr(request, "param", {})
    baseline_models = baseline_redhat_ai_models["api_models"]
    filter_type = param["filter_type"]  # Required parameter

    # Calculate expected models and modify_catalog_source kwargs
    if filter_type == "inclusion":
        expected_models = filter_models_by_pattern(all_models=baseline_models, pattern=param["pattern"])
        modify_kwargs = {"included_models": [param["filter_value"]]}

    elif filter_type == "exclusion":
        models_to_exclude = filter_models_by_pattern(all_models=baseline_models, pattern=param["pattern"])
        expected_models = baseline_models - models_to_exclude
        modify_kwargs = {"excluded_models": [param["filter_value"]]}

    elif filter_type == "combined":
        included_models = filter_models_by_pattern(all_models=baseline_models, pattern=param["include_pattern"])
        expected_models = {model for model in included_models if param["exclude_pattern"] not in model}
        modify_kwargs = {
            "included_models": [param["include_filter_value"]],
            "excluded_models": [param["exclude_filter_value"]],
        }
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")

    # Apply filters
    patch_info = modify_catalog_source(
        admin_client=admin_client, namespace=model_registry_namespace, source_id=REDHAT_AI_CATALOG_ID, **modify_kwargs
    )

    with ResourceEditor(patches={patch_info["configmap"]: patch_info["patch"]}):
        wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

        # Add pod readiness checks if log_cleanup is requested explicitly
        if param.get("log_cleanup", False):
            LOGGER.info(f"Log cleanup: {param['log_cleanup']} requested. Catalog pod would be re-spinned")
            wait_for_model_catalog_pod_ready_after_deletion(
                client=admin_client, model_registry_namespace=model_registry_namespace
            )
        wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)
        yield expected_models

    # Cleanup
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
