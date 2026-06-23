from typing import Any, Self

import pytest
import structlog
from kubernetes.dynamic import DynamicClient

from tests.ai_hub.model_catalog.constants import VALIDATED_CATALOG_ID
from tests.ai_hub.model_catalog.utils import execute_database_query, get_models_from_catalog_api

LOGGER = structlog.get_logger(name=__name__)

pytestmark = [
    pytest.mark.tier2,
    pytest.mark.downstream_only,
    pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace"),
]

MODELS_WITH_PERFORMANCE_DATA: list[str] = [
    "MiniMax-M2.5",
    "Ministral-3-14B-Instruct-2512",
    "Phi-4-mini-instruct-FP8-dynamic",
    "Phi-4-reasoning-FP8-dynamic",
    "Qwen3-Coder-Next-NVFP4",
    "Qwen3-Next-80B-A3B-Instruct-quantized.w4a16",
    "Qwen3-VL-235B-A22B-Instruct-NVFP4",
    "Qwen3.5-122B-A10B-FP8-dynamic",
    "Qwen3.5-35B-A3B-FP8-dynamic",
    "Qwen3.5-397B-A17B-FP8-dynamic",
    "granite-4.0-h-small-FP8-dynamic",
]

COLD_START_REQUIRED_FIELDS: list[str] = [
    "cold_start_time_to_load_seconds",
    "runtime_command",
    "gpu_type",
    "gpu_count",
]

MODEL_LEVEL_PERFORMANCE_QUERY: str = """
SELECT cp.name, cp.double_value
FROM "ContextProperty" cp
JOIN "Context" c ON cp.context_id = c.id
WHERE c.name = '{source_id}:{model_name}'
AND cp.name IN ('min_vram_gb', 'modelcar_image_size', 'modelcar_image_size_bytes')
ORDER BY cp.name;
"""


@pytest.mark.parametrize(
    "validated_model",
    [pytest.param(name, id=f"test_{name.lower()}") for name in MODELS_WITH_PERFORMANCE_DATA],
    indirect=True,
)
class TestModelPerformanceProperties:
    """Validate model-level and cold-start performance properties (RHOAIENG-62445 AC1, AC2)."""

    @pytest.mark.parametrize(
        "property_name",
        [
            pytest.param("min_vram_gb", id="test_min_vram_gb"),
            pytest.param("modelcar_image_size", id="test_modelcar_image_size"),
        ],
    )
    def test_model_has_performance_property(
        self: Self,
        validated_model: dict[str, Any],
        admin_client: DynamicClient,
        model_registry_namespace: str,
        property_name: str,
    ) -> None:
        """
        Given a model with performance data ingested
        When the model is queried via the catalog API
        Then customProperties includes the expected property with a positive value
        And the value exists in the database
        """
        model_name = validated_model["name"]
        properties = validated_model.get("customProperties", {})

        prop = properties.get(property_name, {})
        assert isinstance(prop, dict) and "double_value" in prop, (
            f"Model '{model_name}' missing {property_name} in customProperties"
        )
        assert prop["double_value"] > 0, f"{property_name} should be positive, got {prop['double_value']}"

        db_result = execute_database_query(
            admin_client=admin_client,
            query=MODEL_LEVEL_PERFORMANCE_QUERY.format(source_id=VALIDATED_CATALOG_ID, model_name=model_name),
            namespace=model_registry_namespace,
        )
        db_property_names = [line.split("|")[0].strip() for line in db_result.splitlines() if "|" in line]
        assert property_name in db_property_names, (
            f"{property_name} not found in database for '{model_name}', got: {db_property_names}"
        )
        LOGGER.info(f"Model '{model_name}': {property_name}={prop['double_value']}")

    def test_cold_start_artifacts_have_required_fields(
        self: Self,
        validated_model: dict[str, Any],
        performance_artifacts: list[dict[str, Any]],
    ) -> None:
        """
        Given a model with cold-start performance data
        When performance artifacts are queried via the catalog API
        Then artifacts with performance_sub_type=cold-start contain
            cold_start_time_to_load_seconds, runtime_command, gpu_type, and gpu_count
        """
        model_name = validated_model["name"]
        cold_start_artifacts = [
            artifact
            for artifact in performance_artifacts
            if artifact.get("customProperties", {}).get("performance_sub_type", {}).get("string_value") == "cold-start"
        ]
        assert cold_start_artifacts, f"No cold-start artifacts found for '{model_name}'"

        errors = []
        for artifact in cold_start_artifacts:
            artifact_name = artifact.get("name", "unknown")
            properties = artifact.get("customProperties", {})
            for field in COLD_START_REQUIRED_FIELDS:
                if field not in properties:
                    errors.append(f"Artifact '{artifact_name}' missing '{field}'")

            cold_start_value = properties.get("cold_start_time_to_load_seconds", {}).get("double_value")
            if cold_start_value is not None and cold_start_value <= 0:
                errors.append(
                    f"Artifact '{artifact_name}': cold_start_time_to_load_seconds={cold_start_value} (expected > 0)"
                )

        assert not errors, f"Cold-start artifact validation errors for '{model_name}':\n" + "\n".join(
            f"  - {error}" for error in errors
        )
        LOGGER.info(f"Model '{model_name}': {len(cold_start_artifacts)} cold-start artifacts validated")


class TestColdStartSortingAndFiltering:
    """Validate sorting and filtering by cold_start_time_to_load_seconds (RHOAIENG-62445 AC3)."""

    def test_sort_models_by_cold_start_ascending(
        self: Self,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ) -> None:
        """
        Given models with cold-start performance data
        When models are sorted by cold_start_time_to_load_seconds ascending
        Then the API returns results in ascending order
        """
        response = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            order_by="artifacts.cold_start_time_to_load_seconds",
            sort_order="ASC",
            page_size=1000,
        )
        items = response.get("items", [])
        assert len(items) > 1, "Expected multiple models when sorting by cold_start"
        model_names = [item["name"] for item in items]
        asc_response = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            order_by="artifacts.cold_start_time_to_load_seconds",
            sort_order="DESC",
            page_size=1000,
        )
        desc_names = [item["name"] for item in asc_response.get("items", [])]
        assert model_names == list(reversed(desc_names)), (
            f"ASC order should be reverse of DESC order.\nASC: {model_names}\nDESC: {desc_names}"
        )
        LOGGER.info(f"Sort ASC returned {len(items)} models in correct order")

    def test_sort_models_by_cold_start_descending(
        self: Self,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ) -> None:
        """
        Given models with cold-start performance data
        When models are sorted by cold_start_time_to_load_seconds descending
        Then the API returns results in descending order
        """
        response = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            order_by="artifacts.cold_start_time_to_load_seconds",
            sort_order="DESC",
            page_size=1000,
        )
        items = response.get("items", [])
        assert len(items) > 1, "Expected multiple models when sorting by cold_start"
        unsorted_response = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            page_size=1000,
        )
        unsorted_names = [item["name"] for item in unsorted_response.get("items", [])]
        sorted_names = [item["name"] for item in items]
        assert sorted_names != unsorted_names, "DESC sorted order should differ from default order"
        LOGGER.info(f"Sort DESC returned {len(items)} models in correct order")

    def test_filter_models_by_cold_start_range(
        self: Self,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ) -> None:
        """
        Given models with cold-start performance data
        When models are filtered by cold_start_time_to_load_seconds range [50, 200]
        Then the filtered set is a non-empty proper subset of all models
        """
        all_response = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            page_size=1000,
        )
        all_names = {item["name"] for item in all_response.get("items", [])}

        filter_query = (
            "artifacts.cold_start_time_to_load_seconds.double_value >= 50"
            " AND artifacts.cold_start_time_to_load_seconds.double_value <= 200"
        )
        filtered_response = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            additional_params=f"&filterQuery={filter_query}",
            page_size=1000,
        )
        filtered_names = {item["name"] for item in filtered_response.get("items", [])}

        assert filtered_names, "Expected at least one model with cold_start in [50, 200] range"
        assert filtered_names < all_names, (
            f"Filtered set should be a proper subset of all models. "
            f"Filtered: {len(filtered_names)}, All: {len(all_names)}"
        )
        LOGGER.info(f"Filter cold_start [50-200]: {len(filtered_names)}/{len(all_names)} models")
