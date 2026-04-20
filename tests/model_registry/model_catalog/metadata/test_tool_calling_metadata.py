"""
Tool-calling metadata tests for Model Catalog API (RHAISTRAT-1262, RHOAIENG-53375).

IMPORTANT: These tests validate tool-calling metadata in the model README field, NOT as discrete JSON fields.
"""

from typing import Any, Self

import pytest
from kubernetes.dynamic.exceptions import ResourceNotFoundError

from tests.model_registry.model_catalog.constants import REDHAT_AI_CATALOG_ID, VALIDATED_CATALOG_ID
from tests.model_registry.utils import execute_get_call

pytestmark = [
    pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace", "original_user")
]


class TestToolCallingMetadata:
    """Tests for tool-calling metadata in model README (RHAISTRAT-1262, RHOAIENG-53375)."""

    def test_vllm_tool_calling_deployment_section(
        self: Self,
        tool_calling_model_readme: tuple[str, str],
    ):
        """TC-API-001 & TC-API-002: Verify vLLM tool-calling deployment documentation in README."""
        readme, model_name = tool_calling_model_readme
        errors: list[str] = []

        if "vLLM Deployment with Tool Calling" not in readme:
            errors.append(f"Model '{model_name}' README missing 'vLLM Deployment with Tool Calling' section")

        if "vllm serve" not in readme:
            errors.append(f"Model '{model_name}' README missing 'vllm serve' command")

        if "--tool-call-parser" not in readme:
            errors.append(f"Model '{model_name}' README missing '--tool-call-parser' flag")

        if "Tool Call Parser" not in readme:
            errors.append(f"Model '{model_name}' README missing 'Tool Call Parser' section")

        if model_name not in readme:
            errors.append(f"Model '{model_name}' README does not reference model name")

        if "```bash" not in readme and "```" not in readme:
            errors.append(f"Model '{model_name}' README missing code block formatting")

        assert not errors, f"Model '{model_name}' vLLM deployment issues:\n" + "\n".join(errors)

    def test_auto_tool_choice_flag_in_readme(
        self: Self,
        tool_calling_model_readme: tuple[str, str],
    ):
        """TC-API-003: Verify --enable-auto-tool-choice flag is documented in vLLM command."""
        readme, model_name = tool_calling_model_readme

        assert "--enable-auto-tool-choice" in readme, (
            f"Model '{model_name}' README missing '--enable-auto-tool-choice' flag"
        )

    def test_chat_template_documentation(
        self: Self,
        tool_calling_model_readme: tuple[str, str],
    ):
        """TC-API-005 & TC-API-006: Verify chat template path and file name are documented in README."""
        readme, model_name = tool_calling_model_readme
        errors: list[str] = []

        # TC-API-005: Check that --chat-template flag is present
        if "--chat-template" not in readme:
            errors.append(f"Model '{model_name}' README missing '--chat-template' flag in vLLM command")

        # TC-API-005: Check for "Template path:" documentation
        if "Template path:" not in readme:
            errors.append(f"Model '{model_name}' README missing 'Template path:' documentation")

        # TC-API-006: Check for "Chat template file:" documentation
        if "Chat template file:" not in readme:
            errors.append(f"Model '{model_name}' README missing 'Chat template file:' documentation")

        # TC-API-006: Verify .jinja file is referenced
        if ".jinja" not in readme:
            errors.append(f"Model '{model_name}' README missing .jinja file reference")

        # TC-API-006: Check for Chat Template section header
        if "### Chat Template" not in readme and "## Chat Template" not in readme:
            errors.append(f"Model '{model_name}' README missing 'Chat Template' section header")

        assert not errors, f"Model '{model_name}' chat template issues:\n" + "\n".join(errors)


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
class TestNonToolCallingModelMetadata:
    """Tests for models without tool-calling support (RHAISTRAT-1262)."""

    def test_non_tool_calling_model_readme_without_section(
        self: Self,
        randomly_picked_model_from_catalog_api_by_source: tuple[dict[Any, Any], str, str],
    ):
        """TC-API-004: Verify non-tool-calling model README does not include tool-calling section."""
        model_data, model_name, _ = randomly_picked_model_from_catalog_api_by_source
        readme = model_data["readme"]

        # Skip if model actually supports tool-calling (based on README content)
        if "vLLM Deployment with Tool Calling" in readme:
            pytest.skip(f"Model '{model_name}' has tool-calling section in README; skipping negative test")

        errors: list[str] = []

        # Verify README does not contain tool-calling specific sections
        if "--tool-call-parser" in readme:
            errors.append(
                f"Model '{model_name}' README contains '--tool-call-parser' but should not have tool-calling content"
            )

        if "Tool Call Parser" in readme:
            errors.append(f"Model '{model_name}' README contains 'Tool Call Parser' section but should not")

        assert not errors, f"Model '{model_name}' non-tool-calling README issues:\n" + "\n".join(errors)

    def test_non_tool_calling_model_excludes_auto_tool_choice(
        self: Self,
        randomly_picked_model_from_catalog_api_by_source: tuple[dict[Any, Any], str, str],
    ):
        """TC-API-003: Non-tool-calling model README must not include --enable-auto-tool-choice."""
        model_data, model_name, _ = randomly_picked_model_from_catalog_api_by_source
        readme = model_data["readme"]

        if "vLLM Deployment with Tool Calling" in readme:
            pytest.skip(f"Model '{model_name}' has tool-calling section in README; skipping negative test")

        assert "--enable-auto-tool-choice" not in readme, (
            f"Model '{model_name}' README contains '--enable-auto-tool-choice' flag but should not"
        )


class TestNonExistentModelError:
    """TC-API-007: Error handling for non-existent model requests (RHAISTRAT-1262)."""

    def test_nonexistent_model_returns_error(
        self: Self,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """TC-API-007: Verify non-existent model request returns appropriate error."""
        nonexistent_model_id = "nonexistent-model-v99"
        url = f"{model_catalog_rest_url[0]}sources/{VALIDATED_CATALOG_ID}/models/{nonexistent_model_id}"

        with pytest.raises(ResourceNotFoundError) as exc_info:
            execute_get_call(url=url, headers=model_registry_rest_headers)

        error_message = str(exc_info.value)
        errors: list[str] = []

        if "404" not in error_message:
            errors.append(f"Expected 404 status code in error, got: {error_message}")

        sensitive_patterns = ["stack trace", "psql", "postgres", "password", "secret"]
        for pattern in sensitive_patterns:
            if pattern.lower() in error_message.lower():
                errors.append(f"Error response may contain sensitive information: found '{pattern}'")

        assert not errors, "Non-existent model error issues:\n" + "\n".join(errors)
