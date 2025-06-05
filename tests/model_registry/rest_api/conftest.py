from typing import Any

import pytest

from tests.model_registry.rest_api.constants import MODEL_REGISTRY_BASE_URI
from tests.model_registry.rest_api.utils import register_model_rest_api, execute_model_registry_patch_command
from utilities.constants import Protocols
from utilities.exceptions import MissingParameter


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


@pytest.fixture()
def updated_model_registry_resource(
    request: pytest.FixtureRequest,
    model_registry_rest_url: str,
    model_registry_rest_headers: dict[str, str],
    registered_model_rest_api: dict[str, Any],
) -> dict[str, Any]:
    """
    Generic fixture to update any model registry resource via PATCH request.

    Expects request.param to contain:
        - resource_name: Key to identify the resource in registered_model_rest_api
        - api_name: API endpoint name for the resource type
        - data: JSON data to send in the PATCH request

    Returns:
       Dictionary containing the updated resource data
    """
    resource_name = request.param.get("resource_name")
    api_name = request.param.get("api_name")
    if not (api_name and resource_name):
        raise MissingParameter("resource_name and api_name are required parameters for this fixture.")
    resource_id = registered_model_rest_api[resource_name]["id"]
    assert resource_id, f"Resource id not found: {registered_model_rest_api[resource_name]}"
    return execute_model_registry_patch_command(
        url=f"{model_registry_rest_url}{MODEL_REGISTRY_BASE_URI}{api_name}/{resource_id}",
        headers=model_registry_rest_headers,
        data_json=request.param["data"],
    )
