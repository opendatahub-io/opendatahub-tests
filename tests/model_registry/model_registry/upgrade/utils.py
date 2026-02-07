import pytest
from tests.model_registry.utils import get_and_validate_registered_model
from model_registry import ModelRegistry as ModelRegistryClient
from model_registry.types import RegisteredModel


def validate_upgrade_model_registration(
    model_registry_client: ModelRegistryClient, model_name: str, registered_model: RegisteredModel = None
):
    errors = get_and_validate_registered_model(
        model_registry_client=model_registry_client, model_name=model_name, registered_model=registered_model
    )
    if errors:
        pytest.fail("errors found in model registry response validation:\n{}".format("\n".join(errors)))
