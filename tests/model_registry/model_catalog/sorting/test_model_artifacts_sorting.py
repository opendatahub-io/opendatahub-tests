import pytest
from typing import Self
import random
from simple_logger.logger import get_logger
from tests.model_registry.model_catalog.sorting.utils import (
    get_artifacts_with_sorting,
    validate_items_sorted_correctly,
    verify_custom_properties_sorted,
)
from tests.model_registry.model_catalog.constants import VALIDATED_CATALOG_ID

LOGGER = get_logger(name=__name__)

pytestmark = [pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace")]

MODEL_NAMES_CUSTOM_PROPERTIES: list[str] = [
    "RedHatAI/Llama-3.1-Nemotron-70B-Instruct-HF",
    "RedHatAI/phi-4-quantized.w8a8",
    "RedHatAI/Qwen2.5-7B-Instruct-quantized.w4a16",
]


# More than 1 artifact are available only in downstream
@pytest.mark.downstream_only
class TestArtifactsSorting:
    """Test sorting functionality for GetAllModelArtifacts endpoint
    Fixed on a random model from the validated catalog since we need more than 1 artifact to test sorting.
    """

    @pytest.mark.parametrize(
        "order_by,sort_order,randomly_picked_model_from_catalog_api_by_source",
        [
            (
                "ID",
                "ASC",
                {
                    "catalog_id": VALIDATED_CATALOG_ID,
                    "header_type": "registry",
                    "model_name": random.choice(MODEL_NAMES_CUSTOM_PROPERTIES),
                },
            ),
            (
                "ID",
                "DESC",
                {
                    "catalog_id": VALIDATED_CATALOG_ID,
                    "header_type": "registry",
                    "model_name": random.choice(MODEL_NAMES_CUSTOM_PROPERTIES),
                },
            ),
            pytest.param(
                "NAME",
                "ASC",
                {
                    "catalog_id": VALIDATED_CATALOG_ID,
                    "header_type": "registry",
                    "model_name": random.choice(MODEL_NAMES_CUSTOM_PROPERTIES),
                },
                marks=pytest.mark.xfail(reason="RHOAIENG-38056: falls back to ID sorting"),
            ),
            pytest.param(
                "NAME",
                "DESC",
                {
                    "catalog_id": VALIDATED_CATALOG_ID,
                    "header_type": "registry",
                    "model_name": random.choice(MODEL_NAMES_CUSTOM_PROPERTIES),
                },
                marks=pytest.mark.xfail(reason="RHOAIENG-38056: falls back to ID sorting"),
            ),
        ],
        indirect=["randomly_picked_model_from_catalog_api_by_source"],
    )
    def test_artifacts_sorting_works_correctly(
        self: Self,
        order_by: str,
        sort_order: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        randomly_picked_model_from_catalog_api_by_source: tuple[dict, str, str],
    ):
        """
        RHOAIENG-37260: Test artifacts endpoint sorts correctly by supported fields
        """
        _, model_name, _ = randomly_picked_model_from_catalog_api_by_source
        LOGGER.info(f"Testing artifacts sorting for {model_name}: orderBy={order_by}, sortOrder={sort_order}")

        response = get_artifacts_with_sorting(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_id=VALIDATED_CATALOG_ID,
            model_name=model_name,
            order_by=order_by,
            sort_order=sort_order,
        )

        assert validate_items_sorted_correctly(response["items"], order_by, sort_order)


@pytest.mark.downstream_only
class TestCustomPropertiesSorting:
    """Test sorting functionality for custom properties"""

    @pytest.mark.parametrize(
        "order_by,sort_order,randomly_picked_model_from_catalog_api_by_source,expect_pure_fallback",
        [
            (
                "e2e_p90.double_value",
                "ASC",
                {
                    "catalog_id": VALIDATED_CATALOG_ID,
                    "header_type": "registry",
                    "model_name": random.choice(MODEL_NAMES_CUSTOM_PROPERTIES),
                },
                False,
            ),
            (
                "e2e_p90.double_value",
                "DESC",
                {
                    "catalog_id": VALIDATED_CATALOG_ID,
                    "header_type": "registry",
                    "model_name": random.choice(MODEL_NAMES_CUSTOM_PROPERTIES),
                },
                False,
            ),
            (
                "hardware_count.int_value",
                "ASC",
                {
                    "catalog_id": VALIDATED_CATALOG_ID,
                    "header_type": "registry",
                    "model_name": random.choice(MODEL_NAMES_CUSTOM_PROPERTIES),
                },
                False,
            ),
            (
                "hardware_count.int_value",
                "DESC",
                {
                    "catalog_id": VALIDATED_CATALOG_ID,
                    "header_type": "registry",
                    "model_name": random.choice(MODEL_NAMES_CUSTOM_PROPERTIES),
                },
                False,
            ),
            (
                "hardware_type.string_value",
                "ASC",
                {
                    "catalog_id": VALIDATED_CATALOG_ID,
                    "header_type": "registry",
                    "model_name": random.choice(MODEL_NAMES_CUSTOM_PROPERTIES),
                },
                False,
            ),
            (
                "hardware_type.string_value",
                "DESC",
                {
                    "catalog_id": VALIDATED_CATALOG_ID,
                    "header_type": "registry",
                    "model_name": random.choice(MODEL_NAMES_CUSTOM_PROPERTIES),
                },
                False,
            ),
            (
                "non_existing_property.double_value",
                "ASC",
                {
                    "catalog_id": VALIDATED_CATALOG_ID,
                    "header_type": "registry",
                    "model_name": random.choice(MODEL_NAMES_CUSTOM_PROPERTIES),
                },
                True,
            ),
            (
                "non_existing_property.double_value",
                "DESC",
                {
                    "catalog_id": VALIDATED_CATALOG_ID,
                    "header_type": "registry",
                    "model_name": random.choice(MODEL_NAMES_CUSTOM_PROPERTIES),
                },
                True,
            ),
        ],
        indirect=["randomly_picked_model_from_catalog_api_by_source"],
    )
    def test_custom_properties_sorting_works_correctly(
        self: Self,
        order_by: str,
        sort_order: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        randomly_picked_model_from_catalog_api_by_source: tuple[dict, str, str],
        expect_pure_fallback: bool,
    ):
        """
        RHOAIENG-38010: Test custom properties endpoint sorts correctly by supported fields

        This test validates two scenarios:
        1. expect_pure_fallback=False: Tests custom property sorting where at least some artifacts
           have the property. Items with the property are sorted by the property value (ASC/DESC),
           followed by items without the property sorted by ID ASC (fallback behavior).

        2. expect_pure_fallback=True: Tests pure fallback behavior where NO artifacts have the
           property. All items are sorted by ID ASC, regardless of the requested sortOrder.
        """
        _, model_name, _ = randomly_picked_model_from_catalog_api_by_source
        LOGGER.info(
            f"Testing custom properties sorting for {model_name}: "
            f"orderBy={order_by}, sortOrder={sort_order}, expect_pure_fallback={expect_pure_fallback}"
        )

        response = get_artifacts_with_sorting(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_id=VALIDATED_CATALOG_ID,
            model_name=model_name,
            order_by=order_by,
            sort_order=sort_order,
        )

        # Verify how many artifacts have the custom property
        property_name = order_by.rsplit(".", 1)[0]
        artifacts_with_property = sum(
            1 for item in response["items"] if property_name in item.get("customProperties", {})
        )

        if expect_pure_fallback:
            # When property doesn't exist, sorting always falls back to ID ASC regardless of sortOrder
            assert artifacts_with_property == 0, (
                f"Expected no artifacts to have property {property_name} for pure fallback test, "
                f"but found {artifacts_with_property} artifacts with it"
            )
            is_sorted = validate_items_sorted_correctly(items=response["items"], field="ID", order="ASC")
            assert is_sorted, f"Pure fallback to ID ASC sorting failed for non-existing property {order_by}"
        else:
            # This ensures we're testing actual custom property sorting (not silent fallback)
            assert artifacts_with_property > 0, (
                f"Cannot test custom property sorting: no artifacts have property {property_name}. "
                f"This would result in silent fallback to ID sorting."
            )
            LOGGER.info(f"{artifacts_with_property}/{len(response['items'])} artifacts have property {property_name}")

            # verify_custom_properties_sorted validates:
            # - Items WITH property come first, sorted by property value (respecting sortOrder)
            # - Items WITHOUT property come after, sorted by ID ASC (fallback)
            is_sorted = verify_custom_properties_sorted(
                items=response["items"], property_field=order_by, sort_order=sort_order
            )
            assert is_sorted, f"Custom properties are not sorted correctly for {model_name}"
