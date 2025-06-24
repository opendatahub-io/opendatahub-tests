import pytest
from typing import Self
from tests.model_registry.constants import MODEL_NAME, MODEL_DICT
from model_registry.types import RegisteredModel
from model_registry import ModelRegistry as ModelRegistryClient
from ocp_resources.model_registry_modelregistry_opendatahub_io import ModelRegistry
from simple_logger.logger import get_logger
from tests.model_registry.rest_api.utils import ModelRegistryV1Alpha1

LOGGER = get_logger(name=__name__)


@pytest.mark.parametrize(
    "registered_model",
    [
        pytest.param(
            MODEL_DICT,
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("pre_upgrade_dsc_patch")
class TestPreUpgradeModelRegistry:
    @pytest.mark.pre_upgrade
    def test_registering_model_pre_upgrade(
        self: Self,
        model_registry_client: ModelRegistryClient,
        registered_model: RegisteredModel,
    ):
        model = model_registry_client.get_registered_model(name=MODEL_NAME)
        expected_attrs = {
            "id": registered_model.id,
            "name": registered_model.name,
            "description": registered_model.description,
            "owner": registered_model.owner,
            "state": registered_model.state,
        }
        errors = [
            f"Unexpected {attr} expected: {expected}, received {getattr(model, attr)}"
            for attr, expected in expected_attrs.items()
            if getattr(model, attr) != expected
        ]
        if errors:
            pytest.fail("errors found in model registry response validation:\n{}".format("\n".join(errors)))

    # TODO: if we are in <=2.21, we can create a servicemesh MR here instead of oauth (v1alpha1), and then in
    # post-upgrade check that it automatically gets converted to oauth (v1beta1) - to be done in 2.21 branch directly.


@pytest.mark.usefixtures("post_upgrade_dsc_patch")
class TestPostUpgradeModelRegistry:
    @pytest.mark.post_upgrade
    def test_retrieving_model_post_upgrade(
        self: Self,
        model_registry_client: ModelRegistryClient,
        model_registry_instance: ModelRegistry,
    ):
        model = model_registry_client.get_registered_model(name=MODEL_NAME)
        expected_attrs = {
            "name": MODEL_DICT["model_name"],
        }
        errors = [
            f"Unexpected {attr} expected: {expected}, received {getattr(model, attr)}"
            for attr, expected in expected_attrs.items()
            if getattr(model, attr) != expected
        ]
        if errors:
            LOGGER.error(f"received model: {model}")
            pytest.fail("errors found in model registry response validation:\n{}".format("\n".join(errors)))

        # the following is valid for 2.22+
        api_version = model_registry_instance.instance.apiVersion
        expected_version = f"{ModelRegistry.ApiGroup.MODELREGISTRY_OPENDATAHUB_IO}/{ModelRegistry.ApiVersion.V1BETA1}"
        assert api_version == expected_version

        model_registry_instance_spec = model_registry_instance.instance.spec
        assert not model_registry_instance_spec.istio
        assert model_registry_instance_spec.oauthProxy.serviceRoute == "enabled"

        # After v1alpha1 is removed (2.24?) this has to be removed
        mr_instance = ModelRegistryV1Alpha1(
            name=model_registry_instance.name, namespace=model_registry_instance.namespace, ensure_exists=True
        ).instance
        status = mr_instance.status.to_dict()
        LOGGER.info(f"Validating MR status {status}")
        if not status:
            pytest.fail(f"Empty status found for {mr_instance}")
