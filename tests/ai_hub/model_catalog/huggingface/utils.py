import ast
from typing import Any

import requests
import structlog
from huggingface_hub import HfApi
from kubernetes.dynamic import DynamicClient
from timeout_sampler import retry

from tests.ai_hub.constants import CATALOG_CONTAINER
from tests.ai_hub.model_catalog.constants import HF_SOURCE_ID
from tests.ai_hub.model_catalog.utils import get_models_from_catalog_api
from tests.ai_hub.utils import execute_get_command, get_model_catalog_pod

LOGGER = structlog.get_logger(name=__name__)


def get_huggingface_model_params(model_name: str, huggingface_api: HfApi) -> dict[str, Any]:
    """
    Get some of the fields from HuggingFace API for validation against our model catalog data
    """
    hf_model_info = huggingface_api.model_info(repo_id=model_name)
    fields_mapping = {
        "tags": "tags",
        "gated": "gated",
        "private": "private",
        "architectures": "config.architectures",
        "model_type": "config.model_type",
    }

    result = {}
    for key, path in fields_mapping.items():
        value = get_huggingface_nested_attributes(obj=hf_model_info, attr_path=path)
        if key == "tags":
            value = list(filter(lambda field: not field.startswith("license:"), value))
        # Convert gated to string if it's the gated field
        if key in ["gated", "private"] and value is not None:
            # model registry converts them to lower case. So before validation we need to do the same
            value = str(value).lower()
        result[key] = value
    return result


def get_huggingface_nested_attributes(obj, attr_path) -> Any:
    """
    Get nested attribute using dot notation like 'config.architectures'
    """
    try:
        current_obj = obj
        for index, attr in enumerate(attr_path.split(".")):
            # Handle both object attributes and dictionary keys
            if isinstance(current_obj, dict):
                # For dictionaries, use key access
                if attr not in current_obj:
                    return None
                current_obj = current_obj[attr]
            else:
                # For objects, use attribute access
                if not hasattr(current_obj, attr):
                    return None
                current_obj = getattr(current_obj, attr)
        return current_obj
    except AttributeError as e:
        LOGGER.error(f"AttributeError getting '{attr_path}': {e}")
        return None
    except Exception as e:  # noqa: BLE001
        LOGGER.error(f"Unexpected error getting '{attr_path}': {e}")
        return None


@retry(wait_timeout=60, sleep=5, exceptions_dict={requests.exceptions.ConnectionError: []})
def assert_huggingface_values_matches_model_catalog_api_values(
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    expected_catalog_values: dict[str, str],
    huggingface_api: HfApi,
) -> bool:
    mismatch = {}
    LOGGER.info("Validating HuggingFace model metadata:")
    for model_name in expected_catalog_values:
        url = f"{model_catalog_rest_url[0]}sources/{HF_SOURCE_ID}/models/{model_name}"
        result = execute_get_command(
            url=url,
            headers=model_registry_rest_headers,
        )
        assert result["name"] == model_name
        hf_api_values = get_huggingface_model_params(model_name=model_name, huggingface_api=huggingface_api)
        error = ""
        custom_props = result["customProperties"]
        for field_name in ["gated", "private", "model_type"]:
            prop_key = f"hf_{field_name}"
            if prop_key not in custom_props:
                error += f"Missing expected property '{prop_key}' in customProperties. "
                continue
            model_catalog_value = custom_props[prop_key]["string_value"]
            if model_catalog_value != str(hf_api_values[field_name]):
                error += (
                    f"HuggingFace api value for {field_name} is {hf_api_values[field_name]} and "
                    f"value found from model catalog api call is {model_catalog_value}"
                )
        for field_name in ["architectures", "tags"]:
            prop_key = f"hf_{field_name}"
            if prop_key not in custom_props:
                error += f"Missing expected property '{prop_key}' in customProperties. "
                continue
            raw_value = custom_props[prop_key]["string_value"]
            try:
                field_value = sorted(ast.literal_eval(raw_value))
            except ValueError, SyntaxError:
                error += f"Failed to parse '{prop_key}' value: {raw_value}. "
                continue
            hf_api_value = sorted(hf_api_values[field_name])
            if field_value != hf_api_value:
                error += f"HF api value for {field_name} {field_value} and found {hf_api_value}"
        if error:
            mismatch[model_name] = error

    if mismatch:
        LOGGER.error(f"mismatches are: {mismatch}")
        raise AssertionError("HF api call and model catalog hf models has value mismatch")

    return True


@retry(wait_timeout=60, sleep=5)
def wait_for_huggingface_retrival_match(
    source_id: str,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    expected_num_models_from_hf_api: int,
) -> bool | None:
    # Get all models from the catalog API for the given source
    url = f"{model_catalog_rest_url[0]}models?source={source_id}&pageSize=1000"
    response = execute_get_command(
        url=url,
        headers=model_registry_rest_headers,
    )
    LOGGER.info(f"response: {response['size']}")
    models_response = [model["name"] for model in response["items"]]
    if int(response["size"]) == expected_num_models_from_hf_api:
        LOGGER.info("All models present in the catalog API.")
        return True
    LOGGER.warning(
        f"Expected {expected_num_models_from_hf_api} "
        "models to be present in response. "
        f"Found {response['size']}. Models in "
        f"response: {models_response}"
    )


@retry(wait_timeout=60, sleep=5)
def wait_for_hugging_face_model_import(
    admin_client: DynamicClient, model_registry_namespace: str, hf_id: str, expected_num_models_from_hf_api: int
) -> bool:
    LOGGER.info("Checking pod log for model import information")
    pod = get_model_catalog_pod(client=admin_client, model_registry_namespace=model_registry_namespace)[0]
    log = pod.log(container="catalog")
    if f"{hf_id}: loaded {expected_num_models_from_hf_api} models" in log and f"{hf_id}: cleaned up 0 models" in log:
        LOGGER.info(f"Found log entry confirming model(s) imported for id: {hf_id}")
        return True
    else:
        LOGGER.warning(f"No relevant log entry found: {log}")
        return False


@retry(wait_timeout=60, sleep=5, exceptions_dict={requests.exceptions.ConnectionError: []})
def get_huggingface_model_from_api(
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    model_name: str,
    source_id: str,
) -> dict[str, Any]:
    url = f"{model_catalog_rest_url[0]}sources/{source_id}/models/{model_name}"
    return execute_get_command(
        url=url,
        headers=model_registry_rest_headers,
    )


@retry(wait_timeout=180, sleep=15, print_func_args=False)
def wait_for_last_sync_update_via_logs(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    source_id: str,
    since_seconds: int = 300,
) -> bool:
    """Wait for the catalog pod logs to show a periodic resync for the given source."""
    catalog_pod = get_model_catalog_pod(client=admin_client, model_registry_namespace=model_registry_namespace)[0]
    log = catalog_pod.log(container=CATALOG_CONTAINER, since_seconds=since_seconds)

    sync_marker = f"Periodic sync: reprocessing all models for source {source_id}"
    if sync_marker in log:
        LOGGER.info(f"Periodic sync detected for source {source_id}")
        return True
    return False


@retry(wait_timeout=60, sleep=5, exceptions_dict={requests.exceptions.ConnectionError: [], AssertionError: []})
def wait_for_last_synced_interval_match(
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    model_name: str,
    source_id: str,
    initial_last_synced: str,
    expected_interval: int = 120,
    tolerance: int = 10,
) -> bool:
    """Fetch the model from the catalog API and assert the sync interval is within tolerance.

    Args:
        model_catalog_rest_url: REST URL for model catalog.
        model_registry_rest_headers: Headers for model registry REST API.
        model_name: Name of the model to fetch.
        source_id: Source ID for the catalog.
        initial_last_synced: Initial last_synced epoch-millis value.
        expected_interval: Expected sync interval in seconds.
        tolerance: Allowed deviation in seconds.

    Returns:
        True when the sync interval matches within tolerance.

    Raises:
        AssertionError: If the sync interval difference exceeds the tolerance.
    """
    result = get_huggingface_model_from_api(
        model_catalog_rest_url=model_catalog_rest_url,
        model_registry_rest_headers=model_registry_rest_headers,
        model_name=model_name,
        source_id=source_id,
    )
    current_last_synced = float(result["customProperties"]["last_synced"]["string_value"])
    difference_seconds = (current_last_synced - float(initial_last_synced)) / 1000
    assert abs(difference_seconds - expected_interval) <= tolerance, (
        f"Model {model_name}: expected ~{expected_interval}s sync interval, "
        f"got {difference_seconds:.1f}s. "
        f"Initial: {initial_last_synced}, Current: {current_last_synced}"
    )
    return True


def assert_accessible_models_via_catalog_api(
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    expected_accessible_models: list[str],
    source_label: str | None = None,
    page_size: int = 1000,
):
    """
    Assert that expected accessible models are available through the catalog API.

    Args:
        model_catalog_rest_url: REST URL for model catalog
        model_registry_rest_headers: Headers for model registry REST API
        expected_accessible_models: List of model names that should be accessible
        source_label: Optional source label to filter by (if None, searches all sources)
        page_size: Number of results per page

    Raises:
        AssertionError: If not all expected models are found in the API response
    """
    LOGGER.info(f"Testing catalog API for accessible models: {expected_accessible_models}")
    if source_label:
        LOGGER.info(f"Filtering by source label: {source_label}")
    else:
        LOGGER.info("Searching across all sources (no source label filter)")

    # Get models from catalog API with optional source filtering
    models_response = get_models_from_catalog_api(
        model_catalog_rest_url=model_catalog_rest_url,
        model_registry_rest_headers=model_registry_rest_headers,
        source_label=source_label,
        page_size=page_size,
    )

    available_model_names = [model["name"] for model in models_response.get("items", [])]
    LOGGER.info(f"Models available through catalog API: {available_model_names}")

    missing_models = [model for model in expected_accessible_models if model not in available_model_names]

    assert not missing_models, f"Missing accessible models from catalog API: {missing_models}"
