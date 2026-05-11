"""
Tests for label-based ConfigMap discovery in the model catalog controller (RHOAIENG-46741).

Validates that the catalog controller supports label-based discovery of user-defined
ConfigMaps labeled with opendatahub.io/catalog-source=true.
"""

from typing import Self
from urllib.parse import quote

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap

from tests.model_registry.model_catalog.constants import (
    CATALOG_SOURCE_LABEL_KEY,
    LABELED_SOURCES_PATH_PREFIX,
    REDHAT_AI_CATALOG_ID,
)
from tests.model_registry.model_catalog.labeled_discovery.utils import (
    TEST_MODEL_ALPHA_NAME,
    TEST_MODEL_BETA_NAME,
    TEST_SOURCE_ALPHA_ID,
    TEST_SOURCE_BETA_ID,
    build_labeled_configmap_data,
    get_deployment_catalog_args,
    wait_for_deployment_args_contain,
    wait_for_deployment_args_not_contain,
    wait_for_source_models_loaded,
)
from tests.model_registry.model_catalog.utils import wait_for_model_catalog_api
from tests.model_registry.utils import (
    execute_get_command,
    wait_for_model_catalog_pod_ready_after_deletion,
)

LOGGER = structlog.get_logger(name=__name__)

pytestmark = [
    pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace"),
]


class TestLabeledConfigMapDiscovery:
    """AC1: A labeled ConfigMap is auto-discovered, mounted, and its data is accessible via the catalog API."""

    @pytest.mark.tier1
    def test_labeled_configmap_data_accessible_via_api(
        self: Self,
        labeled_configmap_alpha: ConfigMap,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ) -> None:
        """Verify that a labeled ConfigMap's source appears in the catalog API and its model is queryable.

        Given a ConfigMap with label opendatahub.io/catalog-source=true exists,
        When the operator reconciles and mounts the ConfigMap,
        Then the source appears in the catalog API sources list
        And the model from the ConfigMap is returned when queried by source.
        """
        sources = execute_get_command(
            url=f"{model_catalog_rest_url[0]}sources",
            headers=model_registry_rest_headers,
        )
        source_ids = [source["id"] for source in sources["items"]]
        assert TEST_SOURCE_ALPHA_ID in source_ids, (
            f"Source '{TEST_SOURCE_ALPHA_ID}' not found in catalog sources: {source_ids}"
        )

        models_response = execute_get_command(
            url=f"{model_catalog_rest_url[0]}models?source={TEST_SOURCE_ALPHA_ID}",
            headers=model_registry_rest_headers,
        )
        model_names = [model["name"] for model in models_response["items"]]
        assert TEST_MODEL_ALPHA_NAME in model_names, (
            f"Model '{TEST_MODEL_ALPHA_NAME}' not found in source '{TEST_SOURCE_ALPHA_ID}': {model_names}"
        )

    @pytest.mark.tier1
    def test_labeled_configmap_model_fields_correct(
        self: Self,
        labeled_configmap_alpha: ConfigMap,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ) -> None:
        """Verify model fields from a labeled ConfigMap are correctly populated in the API.

        Given a model is served from a labeled ConfigMap source,
        When the model is queried by source and name,
        Then the returned model has the correct name, provider, source_id, and tasks.
        """
        model = execute_get_command(
            url=(
                f"{model_catalog_rest_url[0]}sources/{TEST_SOURCE_ALPHA_ID}/models/"
                f"{quote(TEST_MODEL_ALPHA_NAME, safe='')}"
            ),
            headers=model_registry_rest_headers,
        )
        assert model["name"] == TEST_MODEL_ALPHA_NAME
        assert model["provider"] == "QA Team"
        assert model["source_id"] == TEST_SOURCE_ALPHA_ID
        assert "text-generation" in model["tasks"]

    @pytest.mark.tier1
    def test_labeled_configmap_deployment_args_updated(
        self: Self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        labeled_configmap_alpha: ConfigMap,
    ) -> None:
        """Verify the model-catalog deployment references the labeled ConfigMap in its --catalogs-path args.

        Given a labeled ConfigMap has been created and reconciled,
        When the deployment container args are inspected,
        Then the labeled source path prefix is present in at least one --catalogs-path arg
        And the specific ConfigMap name appears in the labeled args.
        """
        args = get_deployment_catalog_args(admin_client=admin_client, namespace=model_registry_namespace)
        labeled_args = [arg for arg in args if LABELED_SOURCES_PATH_PREFIX in arg]
        assert labeled_args, f"No labeled source args found in deployment: {args}"
        assert any("test-labeled-alpha" in arg for arg in labeled_args), (
            f"Expected 'test-labeled-alpha' in labeled args: {labeled_args}"
        )


class TestMultipleLabeledConfigMaps:
    """AC2: Multiple labeled ConfigMaps are processed in alphabetical order and both serve data."""

    @pytest.mark.tier1
    def test_multiple_labeled_configmaps_alphabetical_order(
        self: Self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        labeled_configmap_alpha: ConfigMap,
        labeled_configmap_beta: ConfigMap,
    ) -> None:
        """Verify labeled ConfigMaps appear alphabetically in deployment args, after default sources.

        Given two labeled ConfigMaps 'alpha' and 'beta' exist,
        When the deployment args are inspected,
        Then alpha appears before beta in the labeled args
        And all labeled args appear after the default/user-managed source args.
        """
        args = get_deployment_catalog_args(admin_client=admin_client, namespace=model_registry_namespace)
        catalogs_args = [arg for arg in args if "--catalogs-path=" in arg]

        default_indices = [idx for idx, arg in enumerate(catalogs_args) if LABELED_SOURCES_PATH_PREFIX not in arg]
        labeled_indices = [idx for idx, arg in enumerate(catalogs_args) if LABELED_SOURCES_PATH_PREFIX in arg]

        assert default_indices, "No default catalog args found"
        assert len(labeled_indices) >= 2, f"Expected at least 2 labeled args, got: {labeled_indices}"
        assert max(default_indices) < min(labeled_indices), (
            f"Default sources (indices {default_indices}) should come before "
            f"labeled sources (indices {labeled_indices})"
        )

        labeled_args = [catalogs_args[idx] for idx in labeled_indices]
        alpha_idx = next(idx for idx, arg in enumerate(labeled_args) if "test-labeled-alpha" in arg)
        beta_idx = next(idx for idx, arg in enumerate(labeled_args) if "test-labeled-beta" in arg)
        assert alpha_idx < beta_idx, f"Alpha ({alpha_idx}) should come before beta ({beta_idx}) in args: {labeled_args}"

    @pytest.mark.tier1
    def test_both_labeled_configmaps_serve_data_via_api(
        self: Self,
        labeled_configmap_alpha: ConfigMap,
        labeled_configmap_beta: ConfigMap,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ) -> None:
        """Verify both labeled ConfigMaps' sources and models are independently accessible via the API.

        Given two labeled ConfigMaps with distinct sources and models exist,
        When the catalog API is queried,
        Then both source IDs appear in the sources list
        And each source returns its respective model when queried.
        """
        sources = execute_get_command(
            url=f"{model_catalog_rest_url[0]}sources",
            headers=model_registry_rest_headers,
        )
        source_ids = [source["id"] for source in sources["items"]]
        assert TEST_SOURCE_ALPHA_ID in source_ids, f"Alpha source not found: {source_ids}"
        assert TEST_SOURCE_BETA_ID in source_ids, f"Beta source not found: {source_ids}"

        alpha_models = execute_get_command(
            url=f"{model_catalog_rest_url[0]}models?source={TEST_SOURCE_ALPHA_ID}",
            headers=model_registry_rest_headers,
        )
        assert alpha_models["size"] > 0, "No models returned for alpha source"
        assert any(model["name"] == TEST_MODEL_ALPHA_NAME for model in alpha_models["items"])

        beta_models = execute_get_command(
            url=f"{model_catalog_rest_url[0]}models?source={TEST_SOURCE_BETA_ID}",
            headers=model_registry_rest_headers,
        )
        assert beta_models["size"] > 0, "No models returned for beta source"
        assert any(model["name"] == TEST_MODEL_BETA_NAME for model in beta_models["items"])


class TestExistingSourcesUnaffected:
    """AC3: Existing default and user-managed sources continue to work when labeled ConfigMaps exist."""

    @pytest.mark.tier1
    def test_default_sources_still_serve_models(
        self: Self,
        labeled_configmap_alpha: ConfigMap,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ) -> None:
        """Verify default catalog sources remain present and queryable with a labeled ConfigMap active.

        Given a labeled ConfigMap has been created,
        When the catalog API is queried for default sources,
        Then the default Red Hat AI source is present in the sources list
        And the default source still returns models.
        """
        sources = execute_get_command(
            url=f"{model_catalog_rest_url[0]}sources",
            headers=model_registry_rest_headers,
        )
        source_ids = [source["id"] for source in sources["items"]]
        assert REDHAT_AI_CATALOG_ID in source_ids, (
            f"Default source '{REDHAT_AI_CATALOG_ID}' missing after adding labeled ConfigMap: {source_ids}"
        )

        models_response = execute_get_command(
            url=f"{model_catalog_rest_url[0]}models?source={REDHAT_AI_CATALOG_ID}&pageSize=1",
            headers=model_registry_rest_headers,
        )
        assert models_response["items"], (
            f"No models returned from default source '{REDHAT_AI_CATALOG_ID}' after adding labeled ConfigMap"
        )


class TestLabeledConfigMapDeletion:
    """AC4: Deployment updates when a labeled ConfigMap is created or deleted."""

    @pytest.mark.tier1
    def test_labeled_configmap_removal_updates_deployment_and_api(
        self: Self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ) -> None:
        """Verify deleting a labeled ConfigMap removes it from deployment args and the catalog API.

        Given a labeled ConfigMap is created and verified in the deployment and API,
        When the ConfigMap is deleted,
        Then the deployment args no longer reference it
        And the source no longer appears in the catalog API.
        """
        cm_name = "test-labeled-deletion"
        deletion_source_id = "test_deletion_source"
        cm = ConfigMap(
            name=cm_name,
            namespace=model_registry_namespace,
            client=admin_client,
            label={CATALOG_SOURCE_LABEL_KEY: "true"},
            data=build_labeled_configmap_data(
                source_id=deletion_source_id,
                source_name="Test Deletion Source",
                model_name="qa-team/test-deletion-model",
                cm_name=cm_name,
            ),
        )
        with cm:
            LOGGER.info(f"Created labeled ConfigMap for deletion test: {cm_name}")

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
                source_id=deletion_source_id,
            )

            sources = execute_get_command(
                url=f"{model_catalog_rest_url[0]}sources", headers=model_registry_rest_headers
            )
            assert deletion_source_id in [source["id"] for source in sources["items"]]

            cm.delete(wait=True)
            LOGGER.info(f"Deleted labeled ConfigMap: {cm_name}")

            wait_for_deployment_args_not_contain(
                admin_client=admin_client, namespace=model_registry_namespace, unwanted_substring=cm_name
            )
            wait_for_model_catalog_pod_ready_after_deletion(
                client=admin_client, model_registry_namespace=model_registry_namespace
            )
            wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

            sources_after = execute_get_command(
                url=f"{model_catalog_rest_url[0]}sources", headers=model_registry_rest_headers
            )
            assert deletion_source_id not in [source["id"] for source in sources_after["items"]], (
                "Deleted labeled source still appears in catalog API"
            )
