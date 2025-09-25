from model_registry import ModelRegistry as ModelRegistryClient
from model_registry.types import RegisteredModel

from tests.model_registry.constants import MODEL_NAME
from utilities.constants import ModelFormat

MODEL_DESCRIPTION: str = "lorem ipsum"


def get_and_validate_registered_model(
    model_registry_client: ModelRegistryClient,
    model_name: str,
    registered_model: RegisteredModel = None,
) -> list[str]:
    """
    Get and validate a registered model.
    """
    model = model_registry_client.get_registered_model(name=model_name)
    if registered_model is not None:
        expected_attrs = {
            "id": registered_model.id,
            "name": registered_model.name,
            "description": registered_model.description,
            "owner": registered_model.owner,
            "state": registered_model.state,
        }
    else:
        expected_attrs = {
            "name": model_name,
        }
    errors = [
        f"Unexpected {attr} expected: {expected}, received {getattr(model, attr)}"
        for attr, expected in expected_attrs.items()
        if getattr(model, attr) != expected
    ]
    return errors


def register_model(model_registry_client: ModelRegistryClient) -> RegisteredModel:
    return model_registry_client.register_model(
        name=MODEL_NAME,
        uri="https://storage-place.my-company.com",
        version="2.0.0",
        description=MODEL_DESCRIPTION,
        model_format_name=ModelFormat.ONNX,
        model_format_version="1",
        storage_key="my-data-connection",
        storage_path="path/to/model",
        metadata={
            "int_key": 1,
            "bool_key": False,
            "float_key": 3.14,
            "str_key": "str_value",
        },
    )
