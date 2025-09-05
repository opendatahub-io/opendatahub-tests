import pytest
import yaml
from kubernetes.dynamic import DynamicClient

from ocp_resources.deployment import Deployment
from simple_logger.logger import get_logger
from typing import Self, Any

from ocp_resources.pod import Pod
from ocp_resources.config_map import ConfigMap
from ocp_resources.route import Route
from ocp_resources.service import Service
from tests.model_registry.model_catalog.constants import DEFAULT_CATALOG_ID, DEFAULT_CATALOG_NAME
from tests.model_registry.model_catalog.utils import (
    validate_model_catalog_enabled,
    execute_get_command,
    validate_model_catalog_resource,
    validate_default_catalog,
)

LOGGER = get_logger(name=__name__)


@pytest.mark.usefixtures(
    "updated_dsc_component_state_scope_session",
    "model_registry_namespace",
)
class TestModelCatalog:
    @pytest.mark.post_upgrade
    def test_config_map_exists(self: Self, catalog_config_map: ConfigMap):
        # Check that the default configmaps is created when model registry is
        # enabled on data science cluster.
        assert catalog_config_map.exists, f"{catalog_config_map.name} does not exist"
        catalogs = yaml.safe_load(catalog_config_map.instance.data["sources.yaml"])["catalogs"]
        assert catalogs
        assert len(catalogs) == 1, f"{catalog_config_map.name} should have 1 catalog"
        validate_default_catalog(default_catalog=catalogs[0])

    @pytest.mark.parametrize(
        "resource_name",
        [
            pytest.param(
                Deployment,
                id="test_model_catalog_deployment_resource",
            ),
            pytest.param(
                Route,
                id="test_model_catalog_route_resource",
            ),
            pytest.param(
                Service,
                id="test_model_catalog_service_resource",
            ),
            pytest.param(
                Pod,
                id="test_model_catalog_pod_resource",
            ),
        ],
    )
    @pytest.mark.post_upgrade
    def test_model_catalog_resources_exists(
        self: Self, admin_client: DynamicClient, model_registry_namespace: str, resource_name: Any
    ):
        validate_model_catalog_resource(
            kind=resource_name, admin_client=admin_client, namespace=model_registry_namespace
        )

    def test_operator_pod_enabled_model_catalog(self: Self, model_registry_operator_pod: Pod):
        assert validate_model_catalog_enabled(pod=model_registry_operator_pod)

    def test_model_catalog_no_custom_catalog(
        self,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        Validate sources api for model catalog
        """
        result = execute_get_command(
            url=f"{model_catalog_rest_url[0]}sources",
            headers=model_registry_rest_headers,
        )["items"]
        assert result
        assert len(result) == 1, f"Expected no custom models to be present. Actual: {result}"
        assert result[0]["id"] == DEFAULT_CATALOG_ID
        assert result[0]["name"] == DEFAULT_CATALOG_NAME

    def test_default_config_map_not_present(self: Self, model_registry_namespace: str):
        # RHOAIENG-33246: Introduced a new configmap. It should be removed before 2.25 release
        # This test is temporary. So not parameterizing it.
        cfg_map = ConfigMap(name="default-model-catalog", namespace=model_registry_namespace)
        assert not cfg_map.exists, f"{cfg_map.name} should not exist"
