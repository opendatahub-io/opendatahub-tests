import pytest
from typing import Self
from tests.model_registry.constants import MODEL_NAME
from model_registry import ModelRegistry as ModelRegistryClient
from simple_logger.logger import get_logger
from tests.model_registry.utils import get_and_validate_registered_model, register_model

LOGGER = get_logger(name=__name__)


@pytest.mark.usefixtures("pre_upgrade_dsc_patch")
class TestPreUpgradeModelRegistry:
    @pytest.mark.pre_upgrade
    def test_registering_model_pre_upgrade(
        self: Self,
        model_registry_client: ModelRegistryClient,
    ):
        model = register_model(model_registry_client=model_registry_client)
        get_and_validate_registered_model(
            model_registry_client=model_registry_client, model_name=MODEL_NAME, registered_model=model
        )


@pytest.mark.usefixtures("post_upgrade_dsc_patch")
class TestPostUpgradeModelRegistry:
    @pytest.mark.post_upgrade
    def test_retrieving_model_post_upgrade(
        self: Self,
        model_registry_client: ModelRegistryClient,
    ):
        model_registry_client.get_registered_model(name=MODEL_NAME)
