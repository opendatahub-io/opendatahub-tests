from typing import Self, Any
import pytest
from ocp_resources.model_registry_modelregistry_opendatahub_io import ModelRegistry

from tests.model_registry.rest_api.constants import (
    MODEL_REGISTER,
    MODEL_ARTIFACT,
    MODEL_VERSION,
    MODEL_REGISTER_DATA,
    MODEL_ARTIFACT_DESCRIPTION,
    MODEL_FORMAT_NAME,
    MODEL_FORMAT_VERSION,
    MODEL_VERSION_DESCRIPTION,
    STATE_ARCHIVED,
    STATE_LIVE,
    CUSTOM_PROPERTY,
    REGISTERED_MODEL_DESCRIPTION,
)
from tests.model_registry.rest_api.utils import validate_resource_attributes
from simple_logger.logger import get_logger


LOGGER = get_logger(name=__name__)


@pytest.mark.parametrize(
    "model_registry_metadata_db_resources, model_registry_instance, registered_model_rest_api",
    [
        pytest.param(
            {},
            {},
            MODEL_REGISTER_DATA,
            marks=(pytest.mark.smoke),
        ),
        pytest.param(
            {"db_name": "mariadb"},
            {"db_name": "mariadb"},
            MODEL_REGISTER_DATA,
            marks=(pytest.mark.sanity),
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures(
    "updated_dsc_component_state_scope_session",
    "model_registry_namespace",
    "model_registry_metadata_db_resources",
    "model_registry_instance",
    "registered_model_rest_api",
)
@pytest.mark.custom_namespace
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
        validate_resource_attributes(
            expected_params=expected_params,
            actual_resource_data=registered_model_rest_api[data_key],
            resource_name=data_key,
        )

    def test_model_registry_validate_api_version(
        self: Self,
        model_registry_instance,
    ):
        api_version = ModelRegistry(
            name=model_registry_instance[0].name,
            namespace=model_registry_instance[0].namespace,
            ensure_exists=True,
        ).instance.apiVersion
        LOGGER.info(f"Validating apiversion {api_version} for model registry")
        expected_version = f"{ModelRegistry.ApiGroup.MODELREGISTRY_OPENDATAHUB_IO}/{ModelRegistry.ApiVersion.V1BETA1}"
        assert api_version == expected_version

    def test_model_registry_validate_kuberbacproxy_enabled(
        self: Self,
        model_registry_instance,
    ):
        model_registry_instance_spec = model_registry_instance[0].instance.spec
        LOGGER.info(f"Validating that MR is using kubeRBAC proxy {model_registry_instance_spec}")
        assert not model_registry_instance_spec.istio
        assert model_registry_instance_spec.kubeRBACProxy.serviceRoute == "enabled"

    @pytest.mark.parametrize(
        "updated_model_registry_resource, expected_param",
        [
            pytest.param(
                {
                    "resource_name": "model_artifact",
                    "api_name": "model_artifacts",
                    "data": MODEL_ARTIFACT_DESCRIPTION,
                },
                MODEL_ARTIFACT_DESCRIPTION,
                id="test_validate_updated_artifact_description",
            ),
            pytest.param(
                {
                    "resource_name": "model_artifact",
                    "api_name": "model_artifacts",
                    "data": MODEL_FORMAT_NAME,
                },
                MODEL_FORMAT_NAME,
                id="test_validate_updated_artifact_model_format_name",
            ),
            pytest.param(
                {
                    "resource_name": "model_artifact",
                    "api_name": "model_artifacts",
                    "data": MODEL_FORMAT_VERSION,
                },
                MODEL_FORMAT_VERSION,
                id="test_validate_updated_artifact_model_format_version",
            ),
        ],
        indirect=["updated_model_registry_resource"],
    )
    def test_create_update_model_artifact(
        self,
        updated_model_registry_resource: dict[str, Any],
        expected_param: dict[str, Any],
    ):
        """
        Update model artifacts and ensure the updated values are reflected on the artifact
        """

        validate_resource_attributes(
            expected_params=expected_param,
            actual_resource_data=updated_model_registry_resource,
            resource_name="model artifact",
        )

    @pytest.mark.parametrize(
        "updated_model_registry_resource, expected_param",
        [
            pytest.param(
                {
                    "resource_name": "model_version",
                    "api_name": "model_versions",
                    "data": MODEL_VERSION_DESCRIPTION,
                },
                MODEL_VERSION_DESCRIPTION,
                id="test_validate_updated_version_description",
            ),
            pytest.param(
                {"resource_name": "model_version", "api_name": "model_versions", "data": STATE_ARCHIVED},
                STATE_ARCHIVED,
                id="test_validate_updated_version_state_archived",
            ),
            pytest.param(
                {"resource_name": "model_version", "api_name": "model_versions", "data": STATE_LIVE},
                STATE_LIVE,
                id="test_validate_updated_version_state_unarchived",
            ),
            pytest.param(
                {"resource_name": "model_version", "api_name": "model_versions", "data": CUSTOM_PROPERTY},
                CUSTOM_PROPERTY,
                id="test_validate_updated_version_custom_properties",
            ),
        ],
        indirect=["updated_model_registry_resource"],
    )
    def test_updated_model_version(
        self,
        updated_model_registry_resource: dict[str, Any],
        expected_param: dict[str, Any],
    ):
        """
        Update, [RHOAIENG-24371] archive, unarchive model versions and ensure the updated values
        are reflected on the model version
        """
        validate_resource_attributes(
            expected_params=expected_param,
            actual_resource_data=updated_model_registry_resource,
            resource_name="model version",
        )

    @pytest.mark.parametrize(
        "updated_model_registry_resource, expected_param",
        [
            pytest.param(
                {
                    "resource_name": "register_model",
                    "api_name": "registered_models",
                    "data": REGISTERED_MODEL_DESCRIPTION,
                },
                REGISTERED_MODEL_DESCRIPTION,
                id="test_validate_updated_model_description",
            ),
            pytest.param(
                {"resource_name": "register_model", "api_name": "registered_models", "data": STATE_ARCHIVED},
                STATE_ARCHIVED,
                id="test_validate_updated_model_state_archived",
            ),
            pytest.param(
                {"resource_name": "register_model", "api_name": "registered_models", "data": STATE_LIVE},
                STATE_LIVE,
                id="test_validate_updated_model_state_unarchived",
            ),
            pytest.param(
                {"resource_name": "register_model", "api_name": "registered_models", "data": CUSTOM_PROPERTY},
                CUSTOM_PROPERTY,
                id="test_validate_updated_registered_model_custom_properties",
            ),
        ],
        indirect=["updated_model_registry_resource"],
    )
    def test_updated_registered_model(
        self,
        updated_model_registry_resource: dict[str, Any],
        expected_param: dict[str, Any],
    ):
        """
        Update, [RHOAIENG-24371] archive, unarchive registered models and ensure the updated values
        are reflected on the registered model
        """
        validate_resource_attributes(
            expected_params=expected_param,
            actual_resource_data=updated_model_registry_resource,
            resource_name="registered model",
        )
