import re
from typing import Any, Self

import pytest
import structlog

from tests.model_registry.model_catalog.constants import VALIDATED_CATALOG_ID

LOGGER = structlog.get_logger(name=__name__)

VLLM_CONFIG_SECTION_HEADING: str = "vLLM Recommended Configurations"
VLLM_TARGET_MODELS: list[str] = [
    "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",
    "RedHatAI/gpt-oss-120b",
]


@pytest.mark.parametrize(
    "randomly_picked_model_from_catalog_api_by_source",
    [
        pytest.param(
            {"source": VALIDATED_CATALOG_ID, "model_name": model_name, "header_type": "registry"},
            id=model_name.split("/")[-1],
        )
        for model_name in VLLM_TARGET_MODELS
    ],
    indirect=True,
)
class TestVllmConfigEnrichment:
    """Tests for vLLM configuration enrichment in model README (RHOAIENG-53383)."""

    def test_vllm_config_present_in_readme(
        self: Self,
        randomly_picked_model_from_catalog_api_by_source: tuple[dict[Any, Any], str, str],
    ):
        """Verify that target models have vLLM Recommended Configurations section in their README."""
        model_data, model_name, _ = randomly_picked_model_from_catalog_api_by_source
        readme = model_data.get("readme", "")
        assert readme, f"Model '{model_name}' has no readme content"
        assert VLLM_CONFIG_SECTION_HEADING in readme, (
            f"Model '{model_name}' readme does not contain '{VLLM_CONFIG_SECTION_HEADING}' section"
        )

    def test_vllm_config_markdown_structure(
        self: Self,
        randomly_picked_model_from_catalog_api_by_source: tuple[dict[Any, Any], str, str],
    ):
        """Verify that vLLM config markdown follows expected hierarchy and formatting."""
        model_data, model_name, _ = randomly_picked_model_from_catalog_api_by_source
        readme = model_data.get("readme", "")

        # Extract the vLLM config section from the README
        section_start = readme.find(VLLM_CONFIG_SECTION_HEADING)
        assert section_start != -1, f"Model '{model_name}' readme missing '{VLLM_CONFIG_SECTION_HEADING}' section"
        vllm_section = readme[section_start:]

        errors = []

        # Verify code blocks exist (``` markers for vLLM launch commands)
        if "```" not in vllm_section:
            errors.append("No code blocks (```) found in vLLM config section")

        # Verify section has sub-headings (## or ### for presets/modes)
        if not re.search(r"^#{2,4}\s+", vllm_section, re.MULTILINE):
            errors.append("No sub-headings found in vLLM config section")

        # Verify bolded text exists (for hardware or config labels)
        if not re.search(r"\*\*[^*]+\*\*", vllm_section):
            errors.append("No bolded text found in vLLM config section")

        assert not errors, f"Model '{model_name}' vLLM config markdown structure issues:\n" + "\n".join(errors)
