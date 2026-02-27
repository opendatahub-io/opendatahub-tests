from typing import Self

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger

from tests.model_registry.constants import (
    DEFAULT_CUSTOM_MODEL_CATALOG,
    DEFAULT_MODEL_CATALOG_CM,
    MR_INSTANCE_BASE_NAME,
    NUM_RESOURCES,
)
from tests.model_registry.model_registry.rest_api.utils import (
    get_register_model_data,
    register_model_rest_api,
    validate_resource_attributes,
)
from tests.model_registry.utils import get_model_catalog_pod
from utilities.resources.model_registry_modelregistry_opendatahub_io import ModelRegistry

LOGGER = get_logger(name=__name__)


@pytest.mark.parametrize(
    "model_registry_metadata_db_resources, model_registry_instance",
    [
        pytest.param(
            NUM_RESOURCES,
            NUM_RESOURCES,
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures(
    "updated_dsc_component_state_scope_session",
    "model_registry_metadata_db_resources",
    "model_registry_instance",
)
class TestModelRegistryMultipleInstances:
    def test_validate_multiple_model_registry(
        self: Self,
        admin_client: DynamicClient,
        model_registry_instance: list[ModelRegistry],
        model_registry_namespace: str,
    ):
        for num in range(NUM_RESOURCES["num_resources"]):
            mr = ModelRegistry(
                client=admin_client,
                name=f"{MR_INSTANCE_BASE_NAME}{num}",
                namespace=model_registry_namespace,
                ensure_exists=True,
            )
            LOGGER.info(f"{mr.name} found")

    @pytest.mark.tier2
    def test_validate_one_model_catalog_configmap(
        self: Self, admin_client: DynamicClient, model_registry_namespace: str
    ):
        """
        Validate that when multiple MR exists on a cluster, only two model catalog configmaps are created
        """
        expected_number_config_maps: int = 2
        config_map_names = [
            config_map.name
            for config_map in list(ConfigMap.get(namespace=model_registry_namespace, client=admin_client))
            if config_map.name.startswith((DEFAULT_CUSTOM_MODEL_CATALOG, DEFAULT_MODEL_CATALOG_CM))
        ]
        assert len(config_map_names) == expected_number_config_maps, (
            f"Expected {expected_number_config_maps} model catalog sources, found: {config_map_names}"
        )

    def test_validate_model_catalog_pods(self: Self, admin_client: DynamicClient, model_registry_namespace: str):
        """
        Validate that even when multiple MR exists on a cluster, only two model catalog pods are created
        """
        catalog_pods: list[Pod] = get_model_catalog_pod(
            client=admin_client, model_registry_namespace=model_registry_namespace, label_selector="app=model-catalog"
        )
        expected_number_pods: int = 2

        assert len(catalog_pods) == expected_number_pods, (
            f"Expected {expected_number_pods} model catalog pods, found: {[pod.name for pod in catalog_pods]}"
        )

    @pytest.mark.tier2
    def test_validate_register_models_multiple_registries(
        self: Self, model_registry_rest_url: list[str], model_registry_rest_headers: dict[str, str]
    ):
        data = get_register_model_data(num_models=NUM_RESOURCES["num_resources"])
        for num in range(NUM_RESOURCES["num_resources"]):
            result = register_model_rest_api(
                model_registry_rest_url=model_registry_rest_url[num],
                model_registry_rest_headers=model_registry_rest_headers,
                data_dict=data[num],
            )
            for data_key in ["register_model", "model_version", "model_artifact"]:
                validate_resource_attributes(
                    expected_params=data[num][f"{data_key}_data"],
                    actual_resource_data=result[data_key],
                    resource_name=data_key,
                )
