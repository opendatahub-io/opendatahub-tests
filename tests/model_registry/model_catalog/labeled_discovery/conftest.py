from collections.abc import Generator

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap

from tests.model_registry.model_catalog.constants import CATALOG_SOURCE_LABEL_KEY
from tests.model_registry.model_catalog.labeled_discovery.utils import (
    TEST_MODEL_ALPHA_NAME,
    TEST_MODEL_BETA_NAME,
    TEST_SOURCE_ALPHA_ID,
    TEST_SOURCE_BETA_ID,
    build_labeled_configmap_data,
    wait_for_deployment_args_contain,
    wait_for_deployment_args_not_contain,
    wait_for_source_models_loaded,
)
from tests.model_registry.model_catalog.utils import wait_for_model_catalog_api
from tests.model_registry.utils import wait_for_model_catalog_pod_ready_after_deletion

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="class")
def labeled_configmap_alpha(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> Generator[ConfigMap]:
    """Labeled ConfigMap 'alpha' for label-based catalog discovery tests.

    Creates a ConfigMap with the opendatahub.io/catalog-source=true label,
    waits for the operator to reconcile and mount it, then cleans up on teardown.
    """
    cm_name = "test-labeled-alpha"
    cm = ConfigMap(
        name=cm_name,
        namespace=model_registry_namespace,
        client=admin_client,
        label={CATALOG_SOURCE_LABEL_KEY: "true"},
        data=build_labeled_configmap_data(
            source_id=TEST_SOURCE_ALPHA_ID,
            source_name="Test Labeled Alpha",
            model_name=TEST_MODEL_ALPHA_NAME,
            cm_name=cm_name,
        ),
    )
    assert not cm.exists, f"Leftover ConfigMap {cm_name} found — clean up before running tests"

    with cm as created_cm:
        LOGGER.info(f"Created labeled ConfigMap: {cm_name}")
        wait_for_deployment_args_contain(
            admin_client=admin_client, namespace=model_registry_namespace, expected_substring=cm_name
        )
        wait_for_model_catalog_pod_ready_after_deletion(
            client=admin_client, model_registry_namespace=model_registry_namespace
        )
        wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)
        wait_for_source_models_loaded(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_id=TEST_SOURCE_ALPHA_ID,
        )
        yield created_cm

    LOGGER.info(f"Teardown: waiting for deployment to reconcile after deleting {cm_name}")
    wait_for_deployment_args_not_contain(
        admin_client=admin_client, namespace=model_registry_namespace, unwanted_substring=cm_name
    )
    wait_for_model_catalog_pod_ready_after_deletion(
        client=admin_client, model_registry_namespace=model_registry_namespace
    )
    wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)


@pytest.fixture(scope="class")
def labeled_configmap_beta(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> Generator[ConfigMap]:
    """Labeled ConfigMap 'beta' for multi-configmap ordering tests.

    Creates a second labeled ConfigMap to validate alphabetical ordering
    and multi-source support.
    """
    cm_name = "test-labeled-beta"
    cm = ConfigMap(
        name=cm_name,
        namespace=model_registry_namespace,
        client=admin_client,
        label={CATALOG_SOURCE_LABEL_KEY: "true"},
        data=build_labeled_configmap_data(
            source_id=TEST_SOURCE_BETA_ID,
            source_name="Test Labeled Beta",
            model_name=TEST_MODEL_BETA_NAME,
            cm_name=cm_name,
        ),
    )
    assert not cm.exists, f"Leftover ConfigMap {cm_name} found — clean up before running tests"

    with cm as created_cm:
        LOGGER.info(f"Created labeled ConfigMap: {cm_name}")
        wait_for_deployment_args_contain(
            admin_client=admin_client, namespace=model_registry_namespace, expected_substring=cm_name
        )
        wait_for_model_catalog_pod_ready_after_deletion(
            client=admin_client, model_registry_namespace=model_registry_namespace
        )
        wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)
        wait_for_source_models_loaded(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_id=TEST_SOURCE_BETA_ID,
        )
        yield created_cm

    LOGGER.info(f"Teardown: waiting for deployment to reconcile after deleting {cm_name}")
    wait_for_deployment_args_not_contain(
        admin_client=admin_client, namespace=model_registry_namespace, unwanted_substring=cm_name
    )
    wait_for_model_catalog_pod_ready_after_deletion(
        client=admin_client, model_registry_namespace=model_registry_namespace
    )
    wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)
