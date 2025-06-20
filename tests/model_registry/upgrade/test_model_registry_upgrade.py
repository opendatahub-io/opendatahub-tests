import pytest
from typing import Self
from tests.model_registry.constants import MODEL_NAME, MODEL_DICT
from model_registry.types import RegisteredModel
from model_registry import ModelRegistry as ModelRegistryClient
from simple_logger.logger import get_logger

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
        pass

    # TODO: if we are in <=2.21, we can create a servicemesh MR here instead of oauth (v1alpha1), and then in
    # post-upgrade check that it automatically gets converted to oauth (v1beta1)


@pytest.mark.usefixtures("post_upgrade_dsc_patch")
class TestPostUpgradeModelRegistry:
    @pytest.mark.post_upgrade
    def test_retrieving_model_post_upgrade(
        self: Self,
        model_registry_client: ModelRegistryClient,
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

        # TODO: if we are in >= 2.22, we can check that this is using oauth instead of servicemesh
        # TODO: if we are in >= 2.22, we can check that the conversion webhook is working as expected, i.e.
        # the MR instance has api version v1beta1, and that when querying for older api version (e.g.
        # `oc get modelregistries.v1alpha1.modelregistry.opendatahub.io -o wide -n rhoai-model-registries`)
        # it also returns the status stanza (used by dashboard)
