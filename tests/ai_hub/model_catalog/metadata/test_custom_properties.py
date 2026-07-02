from typing import Any

import pytest
import structlog

from tests.ai_hub.model_catalog.constants import REDHAT_AI_CATALOG_ID, VALIDATED_CATALOG_ID
from tests.ai_hub.model_catalog.metadata.utils import (
    extract_custom_property_values,
    get_metadata_from_catalog_pod,
    validate_custom_properties_match_metadata,
)
from tests.ai_hub.utils import execute_get_command_with_retry

LOGGER = structlog.get_logger(name=__name__)

pytestmark = [
    pytest.mark.usefixtures(
        "updated_dsc_component_state_scope_session",
        "model_registry_namespace",
    )
]


@pytest.mark.skip_must_gather
class TestCustomProperties:
    """Test suite for validating custom properties in model catalog API"""

    @pytest.mark.downstream_only
    def test_custom_properties_match_metadata(
        self,
        model_with_benchmark_metadata: tuple[dict[Any, Any], str, str],
        model_catalog_pod,
    ):
        """Test that custom properties from API match values in metadata.json files."""
        model_data, model_name, catalog_id = model_with_benchmark_metadata

        LOGGER.info(f"Testing custom properties metadata match for model '{model_name}' from catalog '{catalog_id}'")

        custom_props = model_data.get("customProperties", {})
        api_props = extract_custom_property_values(custom_properties=custom_props)
        metadata = get_metadata_from_catalog_pod(model_catalog_pod=model_catalog_pod, model_name=model_name)

        assert validate_custom_properties_match_metadata(api_props, metadata)

    @pytest.mark.parametrize("catalog_id", [REDHAT_AI_CATALOG_ID, VALIDATED_CATALOG_ID])
    def test_model_type_field_in_custom_properties(
        self,
        catalog_id: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        Test that all models have model_type with valid values: "generative", "predictive", "unknown".
        """
        valid_model_types = {"generative", "predictive", "unknown"}

        response = execute_get_command_with_retry(
            url=f"{model_catalog_rest_url[0]}models?source={catalog_id}&pageSize=100",
            headers=model_registry_rest_headers,
        )
        models = response["items"]

        LOGGER.info(f"Validating model_type field for {len(models)} models from catalog '{catalog_id}'")

        validation_errors = []

        for model in models:
            custom_properties = model.get("customProperties", {})

            if "model_type" not in custom_properties:
                validation_errors.append(f"Model '{model.get('name')}' missing model_type in customProperties")
                continue

            model_type_value = custom_properties["model_type"]["string_value"]
            if model_type_value not in valid_model_types:
                validation_errors.append(
                    f"Model '{model.get('name')}' has invalid model_type: '{model_type_value}'. "
                    f"Expected one of: {valid_model_types}"
                )

        assert not validation_errors, (
            f"model_type validation failed for {len(validation_errors)} models:\n" + "\n".join(validation_errors)
        )

        LOGGER.info(f"All {len(models)} models in catalog '{catalog_id}' have valid model_type values")


@pytest.mark.downstream_only
@pytest.mark.skip_must_gather
class TestHardwareTagProperty:
    """Tests for RHOAIENG-61492: hardware_tag custom property on Intel Xeon validated models."""

    @pytest.mark.parametrize(
        "model_name",
        [
            pytest.param("RedHatAI/Qwen3-Embedding-8B", id="test_qwen3_embedding_8b"),
            pytest.param("RedHatAI/all-MiniLM-L6-v2", id="test_all_minilm_l6_v2"),
            pytest.param("RedHatAI/embeddinggemma-300m", id="test_embeddinggemma_300m"),
            pytest.param("RedHatAI/granite-embedding-english-r2", id="test_granite_embedding_english_r2"),
            pytest.param("RedHatAI/nomic-embed-text-v1.5", id="test_nomic_embed_text_v1_5"),
            pytest.param(
                "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16",
                id="test_meta_llama_3_1_8b_instruct_w4a16",
            ),
            pytest.param(
                "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
                id="test_meta_llama_3_1_8b_instruct_w8a8",
            ),
        ],
    )
    def test_xeon_model_has_hardware_tag(
        self,
        model_name: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Given an Intel Xeon validated model in the catalog
        When querying its custom properties
        Then hardware_tag should be present with value 'Intel Xeon'
        """
        response = execute_get_command_with_retry(
            url=f"{model_catalog_rest_url[0]}models",
            headers=model_registry_rest_headers,
            params={"pageSize": 1, "filterQuery": f"name='{model_name}'"},
        )
        items = response.get("items", [])
        assert items, f"Model '{model_name}' not found in catalog"

        custom_properties = items[0].get("customProperties", {})
        assert "hardware_tag" in custom_properties, f"Model '{model_name}' missing hardware_tag custom property"
        assert custom_properties["hardware_tag"]["string_value"] == "Intel Xeon", (
            f"Model '{model_name}' hardware_tag is '{custom_properties['hardware_tag']['string_value']}', "
            f"expected 'Intel Xeon'"
        )


MULTILINGUAL_MODELS = [
    pytest.param(
        VALIDATED_CATALOG_ID,
        "RedHatAI/granite-3.1-8b-instruct",
        ["ar", "cs", "de", "en", "es", "fr", "it", "ja", "ko", "nl", "pt", "zh"],
        id="test_granite_3_1_8b_instruct",
    ),
]


@pytest.mark.skip_must_gather
class TestMultilingualModelProperties:
    """Regression tests for RHOAIENG-60065: models with multiple language values must be fully saved."""

    @pytest.mark.parametrize("catalog_id, model_name, expected_languages", MULTILINGUAL_MODELS)
    def test_multilingual_model_has_language_property(
        self,
        catalog_id: str,
        model_name: str,
        expected_languages: list[str],
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Given a model with multiple languages defined in the catalog YAML
        When querying the model via the catalog API
        Then the model should have all expected languages saved
        """
        model = execute_get_command_with_retry(
            url=f"{model_catalog_rest_url[0]}sources/{catalog_id}/models/{model_name}",
            headers=model_registry_rest_headers,
        )
        language = model.get("language", [])
        LOGGER.info(f"Model '{model_name}' language: {language}")
        assert sorted(language) == sorted(expected_languages), (
            f"Model '{model_name}' language mismatch: expected {sorted(expected_languages)}, got {sorted(language)}"
        )
