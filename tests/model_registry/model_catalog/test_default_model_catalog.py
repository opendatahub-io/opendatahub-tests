import pytest
import yaml

from ocp_resources.model_registry_modelregistry_opendatahub_io import ModelRegistry
from simple_logger.logger import get_logger
from typing import Self

from ocp_resources.pod import Pod
from ocp_resources.config_map import ConfigMap
from tests.model_registry.model_catalog.utils import validate_model_catalog_enabled, execute_get_command

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

    def test_config_map_exists(self: Self, model_registry_instance: ModelRegistry, catalog_config_map: ConfigMap):
        # Check that the default configmaps is created when model registry is .
        # Managed configmap contains default model sources while the unmanaged one is empty by default
        assert catalog_config_map.exists, f"{catalog_config_map.name} does not exist"
        models = yaml.safe_load(catalog_config_map.instance.data["sources.yaml"])["catalogs"]
        assert not models, f"Expected no default models to be present. Actual: {models}"

    def test_operator_pod_enabled_model_catalog(
        self: Self, model_registry_instance: ModelRegistry, model_registry_operator_pod: Pod
    ):
        assert validate_model_catalog_enabled(pod=model_registry_operator_pod)

    def test_model_catalog_no_custom_catalog(
        self,
        model_registry_instance: ModelRegistry,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        result = execute_get_command(
            url=f"{model_catalog_rest_url[0]}sources",
            headers=model_registry_rest_headers,
        )["items"]
        assert not result, f"Expected no custom models to be present. Actual: {result}"
