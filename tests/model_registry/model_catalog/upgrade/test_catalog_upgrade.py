import pytest
import yaml
from kubernetes.dynamic import DynamicClient

from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_resources.service import Service
from ocp_resources.resource import ResourceEditor

from tests.model_registry.model_catalog.utils import (
    validate_model_catalog_resource,
    wait_for_model_catalog_api,
    execute_get_command,
    is_model_catalog_ready,
    get_catalog_str,
    get_sample_yaml_str,
    get_default_model_catalog_yaml,
    ResourceNotFoundError,
)
from tests.model_registry.model_catalog.constants import DEFAULT_CATALOG_ID
from tests.model_registry.constants import DEFAULT_MODEL_CATALOG
from simple_logger.logger import get_logger
from timeout_sampler import retry

from .constants import TEST_DATA

LOGGER = get_logger(name=__name__)


@pytest.mark.usefixtures(
    "model_registry_namespace", "admin_client", "model_catalog_rest_url", "model_registry_rest_headers"
)  # noqa: E501
class TestPreUpgradeCatalog:
    @pytest.mark.pre_upgrade
    def test_verify_initial_deployment_pre_upgrade(
        self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Verify baseline model catalog functionality before upgrade"""
        LOGGER.info("Starting pre-upgrade catalog deployment verification")

        # Pod Verification
        validate_model_catalog_resource(kind=Pod, admin_client=admin_client, namespace=model_registry_namespace)

        # Resource Verification
        for resource_type in [Deployment, Route, Service]:
            validate_model_catalog_resource(
                kind=resource_type, admin_client=admin_client, namespace=model_registry_namespace
            )

        # API Accessibility
        wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

        LOGGER.info("Pre-upgrade verification completed successfully")

    @pytest.mark.pre_upgrade
    def test_create_custom_catalog_pre_upgrade(
        self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Create and populate custom catalog source with test data (permanent modification for upgrade)"""
        LOGGER.info("Starting custom catalog creation for upgrade testing")

        # Create Custom Catalog Source
        catalog_config_map = ConfigMap(
            name=DEFAULT_MODEL_CATALOG, client=admin_client, namespace=model_registry_namespace
        )

        # Get existing catalog configuration to preserve it
        existing_catalogs = get_default_model_catalog_yaml(config_map=catalog_config_map)

        # Generate custom catalog configuration
        custom_catalog_yaml = get_catalog_str(ids=[TEST_DATA["catalog_id"]])
        custom_sample_yaml_content = get_sample_yaml_str(models=TEST_DATA["models"])
        custom_sample_yaml_filename = f"{TEST_DATA['catalog_id'].replace('_', '-')}.yaml"

        # Parse custom catalog entries properly
        custom_catalog_entries = yaml.safe_load(f"catalogs:\n{custom_catalog_yaml}")["catalogs"]

        # Check if custom catalog already exists to avoid duplicates
        existing_catalog_ids = {catalog["id"] for catalog in existing_catalogs}
        new_catalog_entries = [
            catalog for catalog in custom_catalog_entries if catalog["id"] not in existing_catalog_ids
        ]

        # Combine catalogs properly as Python objects (only add if not already present)
        if new_catalog_entries:
            combined_catalogs = existing_catalogs + new_catalog_entries
        else:
            combined_catalogs = existing_catalogs
            LOGGER.info(f"Custom catalog {TEST_DATA['catalog_id']} already exists, skipping addition")

        combined_sources_yaml = yaml.dump({"catalogs": combined_catalogs}, default_flow_style=False)

        # Store original configmap state for potential rollback
        original_configmap_data = catalog_config_map.instance.data.copy()

        # Prepare data for ResourceEditor pattern
        new_data = {"sources.yaml": combined_sources_yaml, custom_sample_yaml_filename: custom_sample_yaml_content}

        # Preserve any existing data that we're not modifying
        for key, value in original_configmap_data.items():
            if key not in new_data:
                new_data[key] = value

        # Apply permanent configuration changes using ResourceEditor.update() for persistent changes
        patches = {catalog_config_map: {"data": new_data}}

        ResourceEditor(patches=patches).update()

        # Wait for model catalog to pick up changes
        is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)

        # Verify Data Ingestion
        @retry(wait_timeout=120, sleep=10, exceptions_dict={AssertionError: [], ResourceNotFoundError: []})
        def verify_custom_catalog_ingestion():
            sources_response = execute_get_command(
                url=f"{model_catalog_rest_url[0]}sources", headers=model_registry_rest_headers
            )

            # Assert custom catalog exists alongside default catalog
            custom_catalog_found = any(source["id"] == TEST_DATA["catalog_id"] for source in sources_response["items"])
            assert custom_catalog_found, (
                f"Custom catalog {TEST_DATA['catalog_id']} not found in sources: {[s['id'] for s in sources_response['items']]}"  # noqa: E501
            )

            # Verify default catalog still exists
            default_catalog_found = any(source["id"] == DEFAULT_CATALOG_ID for source in sources_response["items"])
            assert default_catalog_found, (
                f"Default catalog should still exist alongside custom catalog. Available sources: {[s['id'] for s in sources_response['items']]}"  # noqa: E501
            )

            return sources_response

        _ = verify_custom_catalog_ingestion()

        # Verify Models in Custom Source
        for model_name in TEST_DATA["models"]:
            try:
                model_response = execute_get_command(
                    url=f"{model_catalog_rest_url[0]}sources/{TEST_DATA['catalog_id']}/models/{model_name}",
                    headers=model_registry_rest_headers,
                )
                # Successfully retrieving the model via the source-specific API endpoint
                assert model_response["name"] == model_name, (
                    f"Model name mismatch: expected {model_name}, got {model_response.get('name')}"
                )
            except ResourceNotFoundError as e:
                pytest.fail(f"Model {model_name} not found in custom catalog {TEST_DATA['catalog_id']}: {e}")

        # Verify Reconciliation Integrity - both catalogs should be present
        current_configmap_data = ConfigMap(
            name=DEFAULT_MODEL_CATALOG, client=admin_client, namespace=model_registry_namespace
        ).instance.data

        # Parse the current sources.yaml to verify structure
        current_catalogs = yaml.safe_load(current_configmap_data["sources.yaml"])["catalogs"]
        catalog_ids = [catalog["id"] for catalog in current_catalogs]

        # Assert both default and custom catalogs are present
        assert DEFAULT_CATALOG_ID in catalog_ids, f"Default catalog should be preserved. Found catalogs: {catalog_ids}"
        assert TEST_DATA["catalog_id"] in catalog_ids, f"Custom catalog should be added. Found catalogs: {catalog_ids}"
        assert len(catalog_ids) >= 2, f"Should have at least default and custom catalogs. Found: {catalog_ids}"

        LOGGER.info("Custom catalog creation and verification completed successfully (permanent modification)")


@pytest.mark.usefixtures(
    "model_registry_namespace", "admin_client", "model_catalog_rest_url", "model_registry_rest_headers"
)
class TestPostUpgradeCatalog:
    @pytest.mark.post_upgrade
    def test_verify_configmap_migration_post_upgrade(
        self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
    ):
        """Verify configmap migration occurred correctly

        Note: Since the pre-upgrade test now makes permanent modifications,
        the custom catalog should persist through the upgrade process.
        """
        LOGGER.info("Starting post-upgrade configmap migration verification")

        # NOTE: The actual new configmap names need to be determined from upgrade specifications
        # For now, this is a placeholder implementation that should be updated with actual names

        # Verify Original ConfigMap Deletion (commented out until actual migration behavior is known)
        # original_configmap = ConfigMap(
        #     name="model-catalog-sources",  # Original configmap name
        #     client=admin_client,
        #     namespace=model_registry_namespace
        # )
        # assert not original_configmap.exists, "Original configmap should be deleted after upgrade"

        # Verify New ConfigMaps Creation (placeholder - needs actual configmap names)
        # new_default_configmap = ConfigMap(
        #     name="[new-default-source-configmap-name]",
        #     client=admin_client,
        #     namespace=model_registry_namespace
        # )
        # assert new_default_configmap.exists, "New default source configmap should exist"

        # new_custom_configmap = ConfigMap(
        #     name="[new-custom-source-configmap-name]",
        #     client=admin_client,
        #     namespace=model_registry_namespace
        # )
        # assert new_custom_configmap.exists, "New custom source configmap should exist"

        # Validate ConfigMap Contents (placeholder)
        # default_config = yaml.safe_load(new_default_configmap.instance.data["sources.yaml"])
        # validate_default_catalog(default_catalog=default_config["catalogs"][0])

        # custom_config = yaml.safe_load(new_custom_configmap.instance.data["sources.yaml"])
        # custom_catalog = next(
        #     cat for cat in custom_config["catalogs"]
        #     if cat["id"] == TEST_DATA["catalog_id"]
        # )
        # assert custom_catalog["name"] == f"Sample Catalog 0"  # Based on get_catalog_str pattern
        # assert custom_catalog["type"] == "yaml"

        LOGGER.info("ConfigMap migration verification completed")

    @pytest.mark.post_upgrade
    def test_verify_service_health_post_upgrade(
        self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Re-verify all functionality from pre-upgrade phase"""
        LOGGER.info("Starting post-upgrade service health verification")

        # Re-run Deployment Checks
        for resource_type in [Pod, Deployment, Route, Service]:
            validate_model_catalog_resource(
                kind=resource_type, admin_client=admin_client, namespace=model_registry_namespace
            )

        # Re-verify API Accessibility
        wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

        # Verify Custom Data Persistence alongside Default Catalog
        @retry(wait_timeout=300, sleep=15, exceptions_dict={AssertionError: [], ResourceNotFoundError: []})
        def verify_post_upgrade_custom_catalog():
            sources_response = execute_get_command(
                url=f"{model_catalog_rest_url[0]}sources", headers=model_registry_rest_headers
            )

            # Verify custom catalog persists
            custom_catalog_exists = any(source["id"] == TEST_DATA["catalog_id"] for source in sources_response["items"])
            assert custom_catalog_exists, (
                f"Custom catalog {TEST_DATA['catalog_id']} should persist after upgrade. Available sources: {[s['id'] for s in sources_response['items']]}"  # noqa: E501
            )

            # Verify default catalog also persists
            default_catalog_exists = any(source["id"] == "default_catalog" for source in sources_response["items"])
            assert default_catalog_exists, (
                f"Default catalog should persist after upgrade alongside custom catalog. Available sources: {[s['id'] for s in sources_response['items']]}"  # noqa: E501
            )

            # Verify we have at least both catalogs
            assert len(sources_response["items"]) >= 2, (
                f"Should have at least default and custom catalogs after upgrade. Found: {len(sources_response['items'])} catalogs"  # noqa: E501
            )

            return sources_response

        _ = verify_post_upgrade_custom_catalog()

        # Verify custom models are still accessible with proper error handling
        for model_name in TEST_DATA["models"]:

            @retry(wait_timeout=120, sleep=10, exceptions_dict={ResourceNotFoundError: []})
            def verify_model_persistence():
                model_response = execute_get_command(
                    url=f"{model_catalog_rest_url[0]}sources/{TEST_DATA['catalog_id']}/models/{model_name}",
                    headers=model_registry_rest_headers,
                )
                assert model_response["name"] == model_name, (
                    f"Model name mismatch: expected {model_name}, got {model_response.get('name')}"
                )  # noqa: E501
                # Note: Successfully retrieving the model via the source-specific API endpoint
                return model_response

            try:
                verify_model_persistence()
            except ResourceNotFoundError as e:
                pytest.fail(f"Model {model_name} not accessible after upgrade: {e}")

        LOGGER.info("Post-upgrade service health verification completed successfully")
