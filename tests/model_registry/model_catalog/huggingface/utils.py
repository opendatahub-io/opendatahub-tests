import ast
from typing import Any

from tests.model_registry.model_catalog.constants import HF_SOURCE_ID
from tests.model_registry.model_catalog.utils import LOGGER
from tests.model_registry.utils import execute_get_command
from huggingface_hub import HfApi


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
    except Exception as e:
        LOGGER.error(f"Unexpected error getting '{attr_path}': {e}")
        return None


def assert_huggingface_values_matches_model_catalog_api_values(
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    expected_catalog_values: dict[str, str],
    huggingface_api: HfApi,
) -> None:
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
        for field_name in ["gated", "private", "model_type"]:
            model_catalog_value = result["customProperties"][f"hf_{field_name}"]["string_value"]
            if model_catalog_value != str(hf_api_values[field_name]):
                error += (
                    f"HuggingFace api value for {field_name} is {hf_api_values[field_name]} and "
                    f"value found from model catalog api call is {model_catalog_value}"
                )
        for field_name in ["architectures", "tags"]:
            field_value = sorted(ast.literal_eval(result["customProperties"][f"hf_{field_name}"]["string_value"]))
            hf_api_value = sorted(hf_api_values[field_name])
            if field_value != hf_api_value:
                error += f"HF api value for {field_name} {field_value} and found {hf_api_value}"
        if error:
            mismatch[model_name] = error

    if mismatch:
        LOGGER.error(f"mismatches are: {mismatch}")
        raise AssertionError("HF api call and model catalog hf models has value mismatch")
