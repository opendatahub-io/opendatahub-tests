import re
from collections.abc import Generator

import pytest
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import NotFoundError
from ocp_resources.config_map import ConfigMap
from ocp_resources.resource import ResourceEditor
from pytest_testconfig import config as py_config
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler

from tests.model_registry.constants import DEFAULT_CUSTOM_MODEL_CATALOG
from tests.model_registry.model_catalog.catalog_config.utils import (
    filter_models_by_pattern,
    modify_catalog_source,
    wait_for_catalog_source_restore,
)
from tests.model_registry.model_catalog.constants import REDHAT_AI_CATALOG_ID, REDHAT_AI_CATALOG_NAME
from tests.model_registry.utils import (
    get_model_catalog_pod,
    wait_for_model_catalog_api,
    wait_for_model_catalog_pod_ready_after_deletion,
)

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="package")
def recreated_model_catalog_configmap(
    admin_client: DynamicClient,
) -> ConfigMap:
    """
    Package-scoped fixture that deletes the DEFAULT_CUSTOM_MODEL_CATALOG ConfigMap
    and waits for it to be automatically recreated, and cleans up catalog pod, to start with a fresh log

    Returns:
        ConfigMap: The recreated ConfigMap instance
    """
    namespace_name = py_config["model_registry_namespace"]
    # TODO: RHOAIENG-46741 would require changing this to look for configmaps based on label
    # Get the existing ConfigMap
    configmap = ConfigMap(
        name=DEFAULT_CUSTOM_MODEL_CATALOG, client=admin_client, namespace=namespace_name, ensure_exists=True
    )

    LOGGER.info(f"Deleting ConfigMap {DEFAULT_CUSTOM_MODEL_CATALOG} to test recreation")

    # Delete the ConfigMap
    configmap.delete()

    LOGGER.info(f"ConfigMap {DEFAULT_CUSTOM_MODEL_CATALOG} deleted, waiting for recreation")

    # Wait for it to be recreated using TimeoutSampler
    recreated_configmap = ConfigMap(
        name=DEFAULT_CUSTOM_MODEL_CATALOG,
        client=admin_client,
        namespace=namespace_name,
    )

    # Use TimeoutSampler to wait for recreation (2 minutes timeout)
    for sample in TimeoutSampler(
        wait_timeout=120,  # 2 minutes
        sleep=5,
        func=lambda: recreated_configmap.exists,
        exceptions_dict={NotFoundError: []},
    ):
        if sample:  # ConfigMap exists
            break

    LOGGER.info(f"ConfigMap {DEFAULT_CUSTOM_MODEL_CATALOG} recreated successfully")
    wait_for_model_catalog_pod_ready_after_deletion(client=admin_client, model_registry_namespace=namespace_name)
    return recreated_configmap


@pytest.fixture(scope="package", autouse=True)
def catalog_pod_model_counts(
    admin_client: DynamicClient,
    recreated_model_catalog_configmap: ConfigMap,
) -> dict[str, int]:
    """
    Package-scoped auto-use fixture that extracts model counts from catalog pod logs.
    Only applies to tests in catalog_config package.

    Scrapes logs for earliest occurrences of:
    - "redhat_ai_validated_models: loaded x models"
    - "redhat_ai_models: loaded y models"

    Returns:
        Dictionary with keys "redhat_ai_validated_models" and "redhat_ai_models"
        containing the extracted model counts
    """
    # Get the model catalog pod
    namespace_name = py_config["model_registry_namespace"]
    catalog_pods = get_model_catalog_pod(client=admin_client, model_registry_namespace=namespace_name)
    assert len(catalog_pods) > 0, f"No model catalog pods found in namespace {namespace_name}"

    catalog_pod = catalog_pods[0]  # Use the first pod if multiple exist

    # Get pod logs
    logs = catalog_pod.log(container="catalog")

    # Define regex patterns for extraction
    patterns = {
        "redhat_ai_validated_models": r"redhat_ai_validated_models: loaded (\d+) models",
        "redhat_ai_models": r"redhat_ai_models: loaded (\d+) models",
    }

    # Extract counts
    model_counts = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, logs)
        if match:
            model_counts[key] = int(match.group(1))
        else:
            LOGGER.warning(f"Pattern '{pattern}' not found in catalog pod logs")
            model_counts[key] = 0  # Default to 0 if not found

    LOGGER.info(f"Extracted model counts from catalog pod logs: {model_counts}")
    return model_counts


@pytest.fixture(scope="function")
def redhat_ai_models_with_filter(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    baseline_redhat_ai_models: dict[str, set[str] | int],
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    catalog_pod_model_counts: dict[str, int],
) -> Generator[set[str]]:
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
) -> Generator[None]:
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
