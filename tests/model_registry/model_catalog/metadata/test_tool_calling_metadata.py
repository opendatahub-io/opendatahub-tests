from typing import Any, Self

import pytest
import structlog

from tests.model_registry.model_catalog.constants import VALIDATED_CATALOG_ID

LOGGER = structlog.get_logger(name=__name__)

# TODO: Confirm this model name once tool-calling models are ingested into the validated catalog
TOOL_CALLING_TARGET_MODELS: list[str] = [
    "ibm-granite/granite-3.1-8b-instruct",
]

pytestmark = [
    pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace", "original_user")
]


@pytest.mark.parametrize(
    "randomly_picked_model_from_catalog_api_by_source",
    [
        pytest.param(
            {"source": VALIDATED_CATALOG_ID, "model_name": model_name, "header_type": "registry"},
            id=model_name.split("/")[-1],
        )
        for model_name in TOOL_CALLING_TARGET_MODELS
    ],
    indirect=True,
)
class TestToolCallingMetadata:
    """Tests for tool-calling metadata fields in model catalog API (RHAISTRAT-1262)."""

    def test_tool_calling_metadata_fields_present(
        self: Self,
        randomly_picked_model_from_catalog_api_by_source: tuple[dict[Any, Any], str, str],
    ):
        """TC-API-001: Verify that tool-calling metadata fields are present and correctly typed for validated models."""
        model_data, model_name, _ = randomly_picked_model_from_catalog_api_by_source
        errors: list[str] = []

        if "tool_calling_supported" not in model_data:
            errors.append(f"Model '{model_name}' missing 'tool_calling_supported' field")
        elif not isinstance(model_data["tool_calling_supported"], bool):
            errors.append(
                f"Model '{model_name}' field 'tool_calling_supported' is not a bool: "
                f"{type(model_data['tool_calling_supported']).__name__}"
            )
        elif model_data["tool_calling_supported"] is not True:
            errors.append(f"Model '{model_name}' field 'tool_calling_supported' is not True")

        if "required_cli_args" not in model_data:
            errors.append(f"Model '{model_name}' missing 'required_cli_args' field")
        elif not isinstance(model_data["required_cli_args"], list):
            errors.append(
                f"Model '{model_name}' field 'required_cli_args' is not a list: "
                f"{type(model_data['required_cli_args']).__name__}"
            )
        elif not model_data["required_cli_args"]:
            errors.append(f"Model '{model_name}' field 'required_cli_args' is an empty list")

        if "chat_template_path" not in model_data:
            errors.append(f"Model '{model_name}' missing 'chat_template_path' field")
        elif not isinstance(model_data["chat_template_path"], str):
            errors.append(
                f"Model '{model_name}' field 'chat_template_path' is not a str: "
                f"{type(model_data['chat_template_path']).__name__}"
            )
        elif not model_data["chat_template_path"]:
            errors.append(f"Model '{model_name}' field 'chat_template_path' is an empty string")

        if "chat_template_file_name" not in model_data:
            errors.append(f"Model '{model_name}' missing 'chat_template_file_name' field")
        elif not isinstance(model_data["chat_template_file_name"], str):
            errors.append(
                f"Model '{model_name}' field 'chat_template_file_name' is not a str: "
                f"{type(model_data['chat_template_file_name']).__name__}"
            )
        elif not model_data["chat_template_file_name"]:
            errors.append(f"Model '{model_name}' field 'chat_template_file_name' is an empty string")

        if "tool_call_parser" not in model_data:
            errors.append(f"Model '{model_name}' missing 'tool_call_parser' field")
        elif not isinstance(model_data["tool_call_parser"], str):
            errors.append(
                f"Model '{model_name}' field 'tool_call_parser' is not a str: "
                f"{type(model_data['tool_call_parser']).__name__}"
            )
        elif not model_data["tool_call_parser"]:
            errors.append(f"Model '{model_name}' field 'tool_call_parser' is an empty string")

        assert not errors, f"Model '{model_name}' tool-calling metadata issues:\n" + "\n".join(errors)
