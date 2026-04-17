from typing import Any, Self

import pytest
import structlog

from tests.model_registry.model_catalog.constants import REDHAT_AI_CATALOG_ID, VALIDATED_CATALOG_ID

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

    def test_auto_tool_choice_flag_in_api_response(
        self: Self,
        randomly_picked_model_from_catalog_api_by_source: tuple[dict[Any, Any], str, str],
    ):
        """TC-API-003: Verify --enable-auto-tool-choice flag data in BFF API response."""
        model_data, model_name, _ = randomly_picked_model_from_catalog_api_by_source

        assert model_data.get("tool_calling_supported") is True, (
            f"Model '{model_name}' must have tool_calling_supported=True"
        )

        errors: list[str] = []
        auto_tool_choice_flag = "--enable-auto-tool-choice"

        required_cli_args = model_data.get("required_cli_args", [])
        has_flag_in_args = isinstance(required_cli_args, list) and auto_tool_choice_flag in required_cli_args
        has_dedicated_field = "enable_auto_tool_choice" in model_data

        if not has_flag_in_args and not has_dedicated_field:
            errors.append(
                f"Model '{model_name}': '{auto_tool_choice_flag}' not found in "
                f"required_cli_args ({required_cli_args}) or as a dedicated field"
            )

        if has_flag_in_args:
            idx = required_cli_args.index(auto_tool_choice_flag)
            if idx + 1 < len(required_cli_args) and not required_cli_args[idx + 1].startswith("--"):
                errors.append(
                    f"Model '{model_name}': '{auto_tool_choice_flag}' should be a standalone boolean flag, "
                    f"not followed by value '{required_cli_args[idx + 1]}'"
                )

        if has_dedicated_field:
            field_value = model_data["enable_auto_tool_choice"]
            if not isinstance(field_value, bool):
                errors.append(
                    f"Model '{model_name}': 'enable_auto_tool_choice' is not a bool: {type(field_value).__name__}"
                )
            elif field_value is not True:
                errors.append(f"Model '{model_name}': 'enable_auto_tool_choice' is False for a tool-calling model")

        assert not errors, f"Model '{model_name}' auto-tool-choice flag issues:\n" + "\n".join(errors)


@pytest.mark.parametrize(
    "randomly_picked_model_from_catalog_api_by_source",
    [
        pytest.param(
            {"source": REDHAT_AI_CATALOG_ID, "header_type": "registry"},
            id="random-non-tool-calling-model",
        ),
    ],
    indirect=True,
)
class TestNonToolCallingModelExcludesAutoToolChoice:
    """TC-API-003 (step 5): Non-tool-calling model must not include --enable-auto-tool-choice."""

    def test_non_tool_calling_model_excludes_auto_tool_choice(
        self: Self,
        randomly_picked_model_from_catalog_api_by_source: tuple[dict[Any, Any], str, str],
    ):
        """TC-API-003 (step 5): Non-tool-calling model must not include --enable-auto-tool-choice."""
        model_data, model_name, _ = randomly_picked_model_from_catalog_api_by_source

        if model_data.get("tool_calling_supported") is True:
            pytest.skip(f"Model '{model_name}' supports tool-calling; skipping negative test")

        errors: list[str] = []

        required_cli_args = model_data.get("required_cli_args", [])
        if isinstance(required_cli_args, list) and "--enable-auto-tool-choice" in required_cli_args:
            errors.append(
                f"Model '{model_name}' has '--enable-auto-tool-choice' in 'required_cli_args' "
                f"but tool_calling_supported is not True"
            )

        if model_data.get("enable_auto_tool_choice") is True:
            errors.append(
                f"Model '{model_name}' has 'enable_auto_tool_choice' set to True but tool_calling_supported is not True"
            )

        assert not errors, f"Model '{model_name}' auto-tool-choice issues:\n" + "\n".join(errors)
