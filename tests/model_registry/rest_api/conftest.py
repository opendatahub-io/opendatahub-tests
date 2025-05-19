from typing import Any

import pytest

from tests.model_registry.rest_api.constants import MODEL_REGISTRY_BASE_URI
from tests.model_registry.rest_api.utils import register_model_rest_api, execute_model_registry_patch_command
from utilities.constants import Protocols


@pytest.fixture(scope="class")
def model_registry_rest_url(model_registry_instance_rest_endpoint: str) -> str:
    # address and port need to be split in the client instantiation
    return f"{Protocols.HTTPS}://{model_registry_instance_rest_endpoint}"


@pytest.fixture(scope="class")
def model_registry_rest_headers(current_client_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {current_client_token}",
        "accept": "application/json",
        "Content-Type": "application/json",
    }


@pytest.fixture(scope="class")
def registered_model_rest_api(
    request: pytest.FixtureRequest, model_registry_rest_url: str, model_registry_rest_headers: dict[str, str]
) -> dict[str, Any]:
    return register_model_rest_api(
        model_registry_rest_url=model_registry_rest_url,
        model_registry_rest_headers=model_registry_rest_headers,
        data_dict=request.param,
    )


@pytest.fixture(scope="class")
def updated_model_artifact(
    request: pytest.FixtureRequest,
    model_registry_rest_url: str,
    model_registry_rest_headers: dict[str, str],
    registered_model_rest_api: dict[str, Any],
) -> dict[str, Any]:
    model_artifact_id = registered_model_rest_api["model_artifact"]["id"]
    assert model_artifact_id, f"Model artifact id not found: {registered_model_rest_api['model_artifact']}"
    return execute_model_registry_patch_command(
        url=f"{model_registry_rest_url}{MODEL_REGISTRY_BASE_URI}model_artifacts/{model_artifact_id}",
        headers=model_registry_rest_headers,
        data_json=request.param,
    )
