import pytest
import yaml

from class_generator.tests.manifests.Deployment.deployment import Deployment
from ocp_resources.model_registry_modelregistry_opendatahub_io import ModelRegistry
from simple_logger.logger import get_logger
from typing import Self
from kubernetes.dynamic import DynamicClient

from ocp_resources.pod import Pod
from ocp_resources.config_map import ConfigMap
from ocp_resources.service import Service
from tests.model_registry.constants import DEFAULT_MODEL_CATALOG
from tests.model_registry.model_catalog.utils import (
    validate_model_catalog_enabled,
    execute_get_command,
    get_model_catalog_pod,
)

LOGGER = get_logger(name=__name__)


@pytest.mark.usefixtures(
    "updated_dsc_component_state_scope_session",
    "model_registry_namespace",
    "model_registry_metadata_db_resources",
)
class TestModelCatalog:
    def test_config_map_not_created(self: Self, catalog_config_map: ConfigMap):
        # Check that the default configmaps does not exist, when model registry is not created
        assert not catalog_config_map.exists

    def test_config_map_exists(
        self: Self, created_model_registry_for_catalog: ModelRegistry, catalog_config_map: ConfigMap
    ):
        # Check that the default configmaps is created when model registry is enabled.
        assert catalog_config_map.exists, f"{catalog_config_map.name} does not exist"
        models = yaml.safe_load(catalog_config_map.instance.data["sources.yaml"])["catalogs"]
        assert not models, f"Expected no default models to be present. Actual: {models}"

    def test_operator_pod_enabled_model_catalog(
        self: Self, created_model_registry_for_catalog: ModelRegistry, model_registry_operator_pod: Pod
    ):
        assert validate_model_catalog_enabled(pod=model_registry_operator_pod)

    def test_model_catalog_no_custom_catalog(
        self,
        created_model_registry_for_catalog: ModelRegistry,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        result = execute_get_command(
            url=f"{model_catalog_rest_url[0]}sources",
            headers=model_registry_rest_headers,
        )["items"]
        assert not result, f"Expected no custom models to be present. Actual: {result}"

    def test_delete_mr_validate_catalog_config_map(
        self, admin_client: DynamicClient, model_registry_namespace: str, deleted_model_registry_for_catalog: None
    ):
        """
        Ensure that when MR is deleted, MC config map does not get deleted
        """
        config_map = ConfigMap(name=DEFAULT_MODEL_CATALOG, client=admin_client, namespace=model_registry_namespace)
        assert config_map.exists, f"{config_map.name} does not exist"

    def test_delete_mr_validate_catalog_resources(
        self, admin_client: DynamicClient, model_registry_namespace: str, deleted_model_registry_for_catalog: None
    ):
        """
        Ensure that when MR is deleted, MC pod, Deployment, Service gets deleted.
        """
        assert not get_model_catalog_pod(client=admin_client, model_registry_namespace=model_registry_namespace)
        assert not list(
            Deployment.get(
                namespace=model_registry_namespace, label_selector="component=model-catalog", dyn_client=admin_client
            )
        )
        assert not list(
            Service.get(
                namespace=model_registry_namespace, label_selector="component=model-catalog", dyn_client=admin_client
            )
        )
