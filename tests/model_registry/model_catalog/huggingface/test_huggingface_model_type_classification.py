"""
Tests for Hugging Face model_type classification.
"""

from typing import Self

import pytest
from simple_logger.logger import get_logger

from tests.model_registry.model_catalog.utils import get_hf_catalog_str, get_models_from_catalog_api

LOGGER = get_logger(name=__name__)

# Known task types from model-registry hf_catalog.go implementation
# Source: https://github.com/kubeflow/model-registry/blob/main/catalog/internal/catalog/hf_catalog.go
# These task lists are used to classify Hugging Face models as generative, predictive, or unknown
GENERATIVE_TASKS = {
    "text-generation",
    "summarization",
    "translation",
    "text-to-image",
    "unconditional-image-generation",
    "image-to-image",
    "text-to-speech",
    "audio-to-audio",
}

PREDICTIVE_TASKS = {
    "text-classification",
    "image-classification",
    "zero-shot-classification",
    "audio-classification",
    "question-answering",
    "document-question-answering",
    "object-detection",
    "image-segmentation",
    "keypoint-detection",
    "feature-extraction",
    "image-feature-extraction",
    "fill-mask",
}

pytestmark = [pytest.mark.skip_on_disconnected]


@pytest.mark.parametrize(
    "updated_catalog_config_map, hf_source_filter",
    [
        pytest.param(
            {
                "sources_yaml": get_hf_catalog_str(ids=["mixed"]),
            },
            {
                "source_filter": "HuggingFace Source mixed",
            },
            id="test_model_type_classification",
        ),
    ],
    indirect=["updated_catalog_config_map", "hf_source_filter"],
)
class TestModelTypeClassification:
    """RHOAIENG-49090: Test suite for model_type classification on Hugging Face models."""

    @pytest.mark.parametrize("model_type_filter", ["generative", "predictive", "unknown"])
    def test_model_type_filter_and_classification(
        self: Self,
        all_models_unfiltered: dict,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        hf_source_filter: str,
        model_type_filter: str,
    ):
        """
        Verify models are correctly classified and filterable by model_type

        Tests that:
        - Models can be filtered by model_type custom property
        - Filtered models have the expected model_type value
        - Filter returns ALL models with that model_type (no models missed)
        - Unknown models have unrecognized tasks or no tasks
        """
        LOGGER.info(f"Testing {model_type_filter} model classification and filtering")

        # Count models with the expected model_type in unfiltered results
        expected_count = sum(
            model.get("customProperties", {}).get("model_type", {}).get("string_value") == model_type_filter
            for model in all_models_unfiltered["items"]
        )

        LOGGER.info(f"Expected {expected_count} models with model_type='{model_type_filter}' based on unfiltered query")

        # Get filtered results
        filtered_result = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_label=hf_source_filter,
            additional_params=f"&filterQuery=model_type.string_value='{model_type_filter}'",
        )

        LOGGER.info(f"Filter returned {filtered_result['size']} {model_type_filter} models")

        # Verify filter returned all models with that model_type
        assert filtered_result["size"] == expected_count, (
            f"Filter should return {expected_count} models with model_type='{model_type_filter}', "
            f"but returned {filtered_result['size']}"
        )

        # Verify all returned models have the expected model_type
        for model in filtered_result["items"]:
            model_name = model["name"]
            model_type = model["customProperties"]["model_type"]["string_value"]

            # Verify model_type matches filter
            assert model_type == model_type_filter

            # For unknown models, verify they either have no tasks OR unrecognized tasks
            if model_type_filter == "unknown":
                tasks = model.get("tasks", [])
                LOGGER.info(f"Model {model_name} has tasks: {tasks}")

                # If tasks exist, verify none match generative or predictive lists
                if tasks:
                    task_set = set(tasks)
                    recognized_tasks = task_set & (GENERATIVE_TASKS | PREDICTIVE_TASKS)
                    assert not recognized_tasks, (
                        f"Model {model_name} classified as unknown has recognized tasks: {recognized_tasks}"
                    )

            LOGGER.info(f"✓ Model {model_name}: type={model_type}")
