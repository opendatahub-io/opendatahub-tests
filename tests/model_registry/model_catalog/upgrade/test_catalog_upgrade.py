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

        After upgrade, we should have:
        1. model-catalog-sources ConfigMap - contains only custom sources (if any) or empty
        2. model-catalog-default-sources ConfigMap - contains the default sources
        """
        LOGGER.info("Starting post-upgrade configmap migration verification")

        # 1. Verify model-catalog-sources ConfigMap exists and has correct content
        model_catalog_sources_cm = ConfigMap(
            name="model-catalog-sources", client=admin_client, namespace=model_registry_namespace
        )
        assert model_catalog_sources_cm.exists, "model-catalog-sources ConfigMap should exist after upgrade"

        # Parse the sources.yaml content
        sources_data = yaml.safe_load(model_catalog_sources_cm.instance.data["sources.yaml"])

        # Check if it contains only custom sources or is empty
        # If pre-upgrade test created custom sources, they should still be there
        # If no custom sources existed, catalogs should be empty list
        assert "catalogs" in sources_data, "sources.yaml should contain 'catalogs' key"
        catalogs = sources_data["catalogs"]

        # Log what we found for debugging
        catalog_ids = [catalog.get("id", "unknown") for catalog in catalogs]
        LOGGER.info(f"Found catalogs in model-catalog-sources: {catalog_ids}")

        # If our test catalog exists, verify it's properly structured
        test_catalog_found = False
        for catalog in catalogs:
            if catalog.get("id") == TEST_DATA["catalog_id"]:
                test_catalog_found = True
                assert catalog.get("type") == "yaml", f"Custom catalog should be yaml type, got: {catalog.get('type')}"
                assert "name" in catalog, "Custom catalog should have a name"
                break

        # The catalogs list should either be empty or contain only custom catalogs (no default ones)
        if catalogs:
            LOGGER.info(f"Custom catalog {'found' if test_catalog_found else 'not found'} in migrated ConfigMap")
        else:
            LOGGER.info("model-catalog-sources ConfigMap is empty after migration")

        # 2. Verify model-catalog-default-sources ConfigMap exists and has correct content
        model_catalog_default_sources_cm = ConfigMap(
            name="model-catalog-default-sources", client=admin_client, namespace=model_registry_namespace
        )
        assert model_catalog_default_sources_cm.exists, (
            "model-catalog-default-sources ConfigMap should exist after upgrade"
        )

        # Parse and validate the expected default sources content
        default_sources_data = yaml.safe_load(model_catalog_default_sources_cm.instance.data["sources.yaml"])

        # Expected default sources structure
        expected_sources = {
            "redhat_ai_models": {
                "name": "Red Hat AI models",
                "type": "yaml",
                "properties": {"yamlCatalogPath": "/shared-data/models-catalog.yaml"},
            },
            "redhat_ai_validated_models": {
                "name": "Red Hat AI validated models",
                "type": "yaml",
                "properties": {"yamlCatalogPath": "/shared-data/validated-models-catalog.yaml"},
            },
        }

        # Validate that catalogs contain the expected default sources
        assert "catalogs" in default_sources_data, "default sources.yaml should contain 'catalogs' key"
        default_catalogs = default_sources_data["catalogs"]

        # Log what we found for debugging
        default_catalog_ids = [catalog.get("id", "unknown") for catalog in default_catalogs]
        LOGGER.info(f"Found catalogs in model-catalog-default-sources: {default_catalog_ids}")

        # Check for redhat_ai_models source
        redhat_ai_models_found = False
        redhat_ai_validated_models_found = False

        for catalog in default_catalogs:
            catalog_id = catalog.get("id")
            if catalog_id == "redhat_ai_models":
                redhat_ai_models_found = True
                assert catalog["name"] == expected_sources["redhat_ai_models"]["name"], (
                    f"Expected name '{expected_sources['redhat_ai_models']['name']}', got '{catalog.get('name')}'"
                )
                assert catalog["type"] == expected_sources["redhat_ai_models"]["type"], (
                    f"Expected type '{expected_sources['redhat_ai_models']['type']}', got '{catalog.get('type')}'"
                )
                assert (
                    catalog.get("properties", {}).get("yamlCatalogPath")
                    == expected_sources["redhat_ai_models"]["properties"]["yamlCatalogPath"]
                ), (
                    f"Expected yamlCatalogPath '{expected_sources['redhat_ai_models']['properties']['yamlCatalogPath']}', got '{catalog.get('properties', {}).get('yamlCatalogPath')}'"  # noqa: E501
                )
            elif catalog_id == "redhat_ai_validated_models":
                redhat_ai_validated_models_found = True
                assert catalog["name"] == expected_sources["redhat_ai_validated_models"]["name"], (
                    f"Expected name '{expected_sources['redhat_ai_validated_models']['name']}', got '{catalog.get('name')}'"  # noqa: E501
                )
                assert catalog["type"] == expected_sources["redhat_ai_validated_models"]["type"], (
                    f"Expected type '{expected_sources['redhat_ai_validated_models']['type']}', got '{catalog.get('type')}'"  # noqa: E501
                )
                assert (
                    catalog.get("properties", {}).get("yamlCatalogPath")
                    == expected_sources["redhat_ai_validated_models"]["properties"]["yamlCatalogPath"]
                ), (
                    f"Expected yamlCatalogPath '{expected_sources['redhat_ai_validated_models']['properties']['yamlCatalogPath']}', got '{catalog.get('properties', {}).get('yamlCatalogPath')}'"  # noqa: E501
                )

        assert redhat_ai_models_found, (
            f"redhat_ai_models source not found in model-catalog-default-sources. Found: {default_catalog_ids}"
        )
        assert redhat_ai_validated_models_found, (
            f"redhat_ai_validated_models source not found in model-catalog-default-sources. Found: {default_catalog_ids}"  # noqa: E501
        )

        LOGGER.info("ConfigMap migration verification completed successfully")

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
            if resource_type == Pod:
                validate_model_catalog_resource(
                    kind=resource_type,
                    admin_client=admin_client,
                    namespace=model_registry_namespace,
                    expected_resource_count=2,
                )
            else:
                validate_model_catalog_resource(
                    kind=resource_type,
                    admin_client=admin_client,
                    namespace=model_registry_namespace,
                    expected_resource_count=1,
                )

        # Re-verify API Accessibility
        wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

        # Verify Sources Are Available via API
        @retry(wait_timeout=300, sleep=15, exceptions_dict={AssertionError: [], ResourceNotFoundError: []})
        def verify_post_upgrade_sources():
            sources_response = execute_get_command(
                url=f"{model_catalog_rest_url[0]}sources", headers=model_registry_rest_headers
            )

            source_ids = [source["id"] for source in sources_response["items"]]
            LOGGER.info(f"Available sources after upgrade: {source_ids}")

            # Verify the new default sources are accessible via API
            assert "redhat_ai_models" in source_ids, (
                f"redhat_ai_models source not found in API response. Available: {source_ids}"
            )
            assert "redhat_ai_validated_models" in source_ids, (
                f"redhat_ai_validated_models source not found in API response. Available: {source_ids}"
            )

            # If custom catalog was created in pre-upgrade, verify it persists
            custom_catalog_exists = any(source["id"] == TEST_DATA["catalog_id"] for source in sources_response["items"])
            if custom_catalog_exists:
                LOGGER.info(f"Custom catalog {TEST_DATA['catalog_id']} persisted after upgrade")
            else:
                LOGGER.info(
                    f"Custom catalog {TEST_DATA['catalog_id']} not found - may not have been created in pre-upgrade"
                )

            # Verify we have at least the two default sources
            assert len(sources_response["items"]) >= 2, (
                f"Should have at least the two default sources after upgrade. Found: {len(sources_response['items'])} sources"  # noqa: E501
            )

            return sources_response, custom_catalog_exists

        sources_response, custom_catalog_exists = verify_post_upgrade_sources()

        # Verify custom models are still accessible with proper error handling (only if custom catalog exists)
        if custom_catalog_exists:
            LOGGER.info("Verifying custom catalog models are still accessible")
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
        else:
            LOGGER.info("No custom catalog found - skipping custom model verification")

        LOGGER.info("Post-upgrade service health verification completed successfully")
