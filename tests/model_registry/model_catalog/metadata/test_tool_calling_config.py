from typing import Any, Self

import pytest
import structlog

from tests.model_registry.model_catalog.constants import VALIDATED_CATALOG_ID

LOGGER = structlog.get_logger(name=__name__)

TOOL_CALLING_CONFIG_HEADING: str = "Tool Calling Configuration"
TOOL_CALLING_TARGET_MODELS: list[str] = [
    "Qwen/Qwen3-235B-A22B-GPTQ-Int4",
    "openai/gpt-oss-120b",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",  # pragma: allowlist secret
    "mistralai/Mistral-Small-4-119B-2603",
    "Llama-3.3-70B-Instruct",
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
class TestToolCallingConfig:
    """Tests for tool calling configuration in model cards (RHAISTRAT-1473)."""

    def test_enable_auto_tool_choice_in_model_card(
        self: Self,
        randomly_picked_model_from_catalog_api_by_source: tuple[dict[Any, Any], str, str],
    ):
        """TC-CFG-004: Validate --enable-auto-tool-choice flag is documented in model card.

        Confirms that the --enable-auto-tool-choice flag is present in each
        in-scope model card's Tool Calling Configuration section.
        """
        model_data, model_name, _ = randomly_picked_model_from_catalog_api_by_source
        readme = model_data.get("readme", "")
        assert readme, f"Model '{model_name}' has no readme content"
        assert TOOL_CALLING_CONFIG_HEADING in readme, (
            f"Model '{model_name}' readme does not contain '{TOOL_CALLING_CONFIG_HEADING}' section"
        )

        section_start = readme.find(TOOL_CALLING_CONFIG_HEADING)
        tool_calling_section = readme[section_start:]

        assert "--enable-auto-tool-choice" in tool_calling_section, (
            f"Model '{model_name}' Tool Calling Configuration section does not contain '--enable-auto-tool-choice' flag"
        )
