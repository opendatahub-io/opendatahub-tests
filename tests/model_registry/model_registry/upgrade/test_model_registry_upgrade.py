import pytest
from typing import Self, Any


from tests.model_registry.constants import MODEL_NAME, MODEL_DICT
from model_registry.types import RegisteredModel
from model_registry import ModelRegistry as ModelRegistryClient

from tests.model_registry.model_registry.upgrade.utils import validate_upgrade_model_registration
from utilities.constants import ModelFormat
from utilities.resources.model_registry_modelregistry_opendatahub_io import ModelRegistry
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)
MODEL_NAME_DEFAULT_DB: str = f"{MODEL_NAME}-default-db"
MODEL_DICT_DEFAULT_DB: dict[str, Any] = {
    "model_name": MODEL_NAME_DEFAULT_DB,
    "model_uri": "https://storage-place.my-company.com",
    "model_version": "2.0.1",
    "model_description": "My description",
    "model_format": ModelFormat.ONNX,
    "model_format_version": "1.1",
    "model_storage_key": "my-data-connection1",
    "model_storage_path": "path/to/model1",
    "model_metadata": {
        "int_key": 1,
        "bool_key": False,
        "float_key": 3.145,
        "str_key": "str_value",
    },
}


@pytest.mark.parametrize(
    "registered_model, registered_model_default_db",
    [
        pytest.param(MODEL_DICT, MODEL_DICT_DEFAULT_DB),
    ],
    indirect=True,
)
@pytest.mark.usefixtures(
    "model_registry_metadata_db_resources",
    "model_registry_instance",
    "model_registry_instance_default_db",
    "model_registry_default_db_instance_rest_endpoint",
    "model_registry_client_default_db",
    "registered_model",
    "registered_model_default_db",
)
class TestPreUpgradeModelRegistry:
    @pytest.mark.pre_upgrade
    def test_registering_model_pre_upgrade_mysql(
        self: Self,
        model_registry_client: list[ModelRegistryClient],
        registered_model: RegisteredModel,
    ):
        validate_upgrade_model_registration(
            model_registry_client=model_registry_client[0], model_name=MODEL_NAME, registered_model=registered_model
        )

    @pytest.mark.pre_upgrade
    def test_registering_model_default_db_pre_upgrade(
        self: Self,
        model_registry_instance_default_db: ModelRegistry,
        model_registry_client_default_db: list[ModelRegistryClient],
        registered_model_default_db: RegisteredModel,
    ):
        validate_upgrade_model_registration(
            model_registry_client=model_registry_client_default_db,
            model_name=MODEL_NAME_DEFAULT_DB,
            registered_model=registered_model_default_db,
        )


class TestPostUpgradeModelRegistry:
    @pytest.mark.post_upgrade
    def test_retrieving_model_post_upgrade_mysql(
        self: Self,
        model_registry_client: list[ModelRegistryClient],
        model_registry_instance: list[ModelRegistry],
        model_registry_instance_default_db: ModelRegistry,
    ):
        validate_upgrade_model_registration(model_registry_client=model_registry_client[0], model_name=MODEL_NAME)

    @pytest.mark.post_upgrade
    def test_retrieving_model_default_db_post_upgrade(
        self: Self,
        model_registry_client_default_db: ModelRegistryClient,
        model_registry_instance_default_db: ModelRegistry,
    ):
        validate_upgrade_model_registration(
            model_registry_client=model_registry_client_default_db, model_name=MODEL_NAME_DEFAULT_DB
        )

    @pytest.mark.post_upgrade
    def test_model_registry_instance_api_version_post_upgrade(
        self: Self,
        model_registry_instance,
    ):
        # the following is valid for 2.22+
        api_version = model_registry_instance[0].instance.apiVersion
        expected_version = f"{ModelRegistry.ApiGroup.MODELREGISTRY_OPENDATAHUB_IO}/{ModelRegistry.ApiVersion.V1BETA1}"
        assert api_version == expected_version

    @pytest.mark.post_upgrade
    def test_model_registry_instance_spec_post_upgrade(
        self: Self,
        model_registry_instance,
    ):
        model_registry_instance_spec = model_registry_instance[0].instance.spec
        assert not model_registry_instance_spec.istio
        assert model_registry_instance_spec.kubeRBACProxy.serviceRoute == "enabled"
