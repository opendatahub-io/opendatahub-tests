from typing import Any
import requests
import json
from simple_logger.logger import get_logger
from tests.model_registry.exceptions import (
    ModelRegistryResourceNotCreated,
    ModelRegistryResourceNotFoundError,
    ModelRegistryResourceNotUpdated,
)
from tests.model_registry.rest_api.constants import MODEL_REGISTRY_BASE_URI

LOGGER = get_logger(name=__name__)


def execute_model_registry_patch_command(
    url: str, headers: dict[str, str], data_json: dict[str, Any]
) -> dict[Any, Any]:
    resp = requests.patch(url=url, json=data_json, headers=headers, verify=False, timeout=60)
    LOGGER.info(f"url: {url}, status code: {resp.status_code}, rep: {resp.text}")

    if resp.status_code != 200:
        raise ModelRegistryResourceNotUpdated(
            f"Failed to update ModelRegistry resource: {url}, {resp.status_code}: {resp.text}"
        )
    try:
        return json.loads(resp.text)
    except json.JSONDecodeError:
        LOGGER.error(f"Unable to parse {resp.text}")
        raise


def execute_model_registry_post_command(url: str, headers: dict[str, str], data_json: dict[str, Any]) -> dict[Any, Any]:
    resp = requests.post(url=url, json=data_json, headers=headers, verify=False, timeout=60)
    LOGGER.info(f"url: {url}, status code: {resp.status_code}, rep: {resp.text}")

    if resp.status_code not in [200, 201]:
        raise ModelRegistryResourceNotCreated(
            f"Failed to create ModelRegistry resource: {url}, {resp.status_code}: {resp.text}"
        )
    try:
        return json.loads(resp.text)
    except json.JSONDecodeError:
        LOGGER.error(f"Unable to parse {resp.text}")
        raise


def execute_model_registry_get_command(url: str, headers: dict[str, str]) -> dict[Any, Any]:  # skip-unused-code
    resp = requests.get(url=url, headers=headers, verify=False)
    LOGGER.info(f"url: {url}, status code: {resp.status_code}, rep: {resp.text}")
    if resp.status_code not in [200, 201]:
        raise ModelRegistryResourceNotFoundError(
            f"Failed to get ModelRegistry resource: {url}, {resp.status_code}: {resp.text}"
        )

    try:
        return json.loads(resp.text)
    except json.JSONDecodeError:
        LOGGER.error(f"Unable to parse {resp.text}")
        raise


def register_model_rest_api(
    model_registry_rest_url: str, model_registry_rest_headers: dict[str, str], data_dict: dict[str, Any]
) -> dict[str, Any]:
    # register a model
    register_model = execute_model_registry_post_command(
        url=f"{model_registry_rest_url}{MODEL_REGISTRY_BASE_URI}registered_models",
        headers=model_registry_rest_headers,
        data_json=data_dict["register_model_data"],
    )
    # create associated model version:
    model_data = data_dict["model_version_data"]
    model_data["registeredModelId"] = register_model["id"]
    model_version = execute_model_registry_post_command(
        url=f"{model_registry_rest_url}{MODEL_REGISTRY_BASE_URI}model_versions",
        headers=model_registry_rest_headers,
        data_json=model_data,
    )
    # create associated model artifact
    model_artifact = execute_model_registry_post_command(
        url=f"{model_registry_rest_url}{MODEL_REGISTRY_BASE_URI}model_versions/{model_version['id']}/artifacts",
        headers=model_registry_rest_headers,
        data_json=data_dict["model_artifact_data"],
    )
    LOGGER.info(
        f"Successfully registered model: {register_model}, with version: {model_version} and "
        f"associated artifact: {model_artifact}"
    )
    return {"register_model": register_model, "model_version": model_version, "model_artifact": model_artifact}
