from typing import Self, Any
import pytest
from pytest_testconfig import config as py_config
from ocp_resources.model_registry_modelregistry_opendatahub_io import ModelRegistry

from tests.model_registry.constants import MR_INSTANCE_BASE_NAME
from tests.model_registry.rest_api.utils import (
    validate_resource_attributes,
    get_register_model_data,
    register_model_rest_api,
)
from utilities.constants import DscComponents
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)
NUM_RESOURCES = {"num": 3}


@pytest.mark.parametrize(
    "updated_dsc_component_state_scope_class, model_registry_db_secret, model_registry_db_pvc, "
    "model_registry_db_service, model_registry_db_deployment, model_registry_instance_mysql",
    [
        pytest.param(
            {
                "component_patch": {
                    DscComponents.MODELREGISTRY: {
                        "managementState": DscComponents.ManagementState.MANAGED,
                        "registriesNamespace": py_config["model_registry_namespace"],
                    },
                },
            },
            NUM_RESOURCES,
            NUM_RESOURCES,
            NUM_RESOURCES,
            NUM_RESOURCES,
            NUM_RESOURCES,
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures(
    "updated_dsc_component_state_scope_class",
    "model_registry_db_secret",
    "model_registry_db_pvc",
    "model_registry_db_service",
    "model_registry_db_deployment",
    "model_registry_instance_mysql",
)
class TestModelRegistryMultipleInstances:
    def test_validate_multiple_model_registry(
        self: Self, model_registry_instance_mysql: list[Any], model_registry_namespace: str
    ):
        for num in range(0, NUM_RESOURCES["num"]):
            mr = ModelRegistry(
                name=f"{MR_INSTANCE_BASE_NAME}{num}", namespace=model_registry_namespace, ensure_exists=True
            )
            LOGGER.info(f"{mr.name} found")

    def test_validate_register_models_multiple_registries(
        self: Self, model_registry_rest_url: list[str], model_registry_rest_headers: dict[str, str]
    ):
        data = get_register_model_data(num_models=NUM_RESOURCES["num"])
        for num in range(0, NUM_RESOURCES["num"]):
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
