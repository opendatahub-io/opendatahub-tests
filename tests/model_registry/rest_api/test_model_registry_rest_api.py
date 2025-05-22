from typing import Self, Any
import pytest
from pytest_testconfig import config as py_config

from tests.model_registry.rest_api.constants import MODEL_REGISTER, MODEL_ARTIFACT, MODEL_VERSION, MODEL_REGISTER_DATA
from utilities.constants import DscComponents
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)


@pytest.mark.parametrize(
    "updated_dsc_component_state_scope_class, registered_model_rest_api",
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
            MODEL_REGISTER_DATA,
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("updated_dsc_component_state_scope_class", "registered_model_rest_api")
class TestModelRegistryCreationRest:
    """
    Tests the creation of a model registry. If the component is set to 'Removed' it will be switched to 'Managed'
    for the duration of this test module.
    """

    @pytest.mark.parametrize(
        "expected_params, data_key",
        [
            pytest.param(
                MODEL_REGISTER,
                "register_model",
                id="test_validate_registered_model",
            ),
            pytest.param(
                MODEL_VERSION,
                "model_version",
                id="test_validate_model_version",
            ),
            pytest.param(
                MODEL_ARTIFACT,
                "model_artifact",
                id="test_validate_model_artifact",
            ),
        ],
    )
    def test_validate_model_registry_resource(
        self: Self,
        registered_model_rest_api: dict[str, Any],
        expected_params: dict[str, str],
        data_key: str,
    ):
        errors: list[str:Any]
        created_resource_data = registered_model_rest_api[data_key]
        if errors := [
            {key: [expected_params[key], created_resource_data.get(key)]}
            for key in expected_params.keys()
            if not created_resource_data.get(key) or created_resource_data[key] != expected_params[key]
        ]:
            pytest.fail(f"Model did not get created with expected values: {errors}")
        LOGGER.info(f"Successfully validated: {created_resource_data['name']}")

    @pytest.mark.parametrize(
        "updated_model_artifact, expected_param",
        [
            pytest.param(
                {"description": "updated description"},
                {"description": "updated description"},
                id="test_validate_updated_artifact_description",
            ),
            pytest.param(
                {"modelFormatName": "tensorflow"},
                {"modelFormatName": "tensorflow"},
                id="test_validate_updated_artifact_model_format_name",
            ),
            pytest.param(
                {"modelFormatVersion": "v2"},
                {"modelFormatVersion": "v2"},
                id="test_validate_updated_artifact_model_format_version",
            ),
        ],
        indirect=["updated_model_artifact"],
    )
    def test_create_update_model_artifact(
        self,
        updated_model_artifact: dict[str, Any],
        expected_param: dict[str, Any],
    ):
        errors: list[dict[str, list[Any]]]
        if errors := [
            {key: [expected_param[key], updated_model_artifact.get(key)]}
            for key in expected_param.keys()
            if not updated_model_artifact.get(key) or updated_model_artifact[key] != expected_param[key]
        ]:
            pytest.fail(f"Model did not get updated with expected values: {errors}")
        LOGGER.info(f"Successfully validated: {updated_model_artifact['name']}")
