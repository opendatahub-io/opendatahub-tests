import pytest
from typing import Self
from ocp_resources.config_map import ConfigMap
from simple_logger.logger import get_logger
from tests.model_registry.model_catalog.utils import (
    get_models_from_catalog_api,
    get_sources_with_sorting,
    get_artifacts_with_sorting,
    validate_items_sorted_correctly,
    verify_custom_properties_sorted,
)
from tests.model_registry.model_catalog.constants import VALIDATED_CATALOG_ID

LOGGER = get_logger(name=__name__)

pytestmark = [pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace")]


class TestModelsSorting:
    """Test sorting functionality for FindModels endpoint"""

    @pytest.mark.parametrize(
        "order_by,sort_order",
        [
            ("ID", "ASC"),
            ("ID", "DESC"),
            pytest.param(
                "NAME",
                "ASC",
                marks=pytest.mark.xfail(
                    reason="RHOAIENG-38056: Backend bug - NAME sorting not implemented, falls back to ID sorting"
                ),
            ),
            pytest.param(
                "NAME",
                "DESC",
                marks=pytest.mark.xfail(
                    reason="RHOAIENG-38056: Backend bug - NAME sorting not implemented, falls back to ID sorting"
                ),
            ),
            ("CREATE_TIME", "ASC"),
            ("CREATE_TIME", "DESC"),
            ("LAST_UPDATE_TIME", "ASC"),
            ("LAST_UPDATE_TIME", "DESC"),
        ],
    )
    def test_models_sorting_works_correctly(
        self: Self,
        enabled_model_catalog_config_map: ConfigMap,
        order_by: str,
        sort_order: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        RHOAIENG-37260: Test models endpoint sorts correctly by field and order
        """
        LOGGER.info(f"Testing models sorting: orderBy={order_by}, sortOrder={sort_order}")

        response = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            order_by=order_by,
            sort_order=sort_order,
        )

        assert validate_items_sorted_correctly(response["items"], order_by, sort_order)


class TestSourcesSorting:
    """Test sorting functionality for FindSources endpoint"""

    @pytest.mark.parametrize(
        "order_by,sort_order",
        [
            ("ID", "ASC"),
            ("ID", "DESC"),
            ("NAME", "ASC"),
            ("NAME", "DESC"),
        ],
    )
    def test_sources_sorting_works_correctly(
        self: Self,
        enabled_model_catalog_config_map: ConfigMap,
        order_by: str,
        sort_order: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        RHOAIENG-37260: Test sources endpoint sorts correctly by supported fields
        """
        LOGGER.info(f"Testing sources sorting: orderBy={order_by}, sortOrder={sort_order}")

        response = get_sources_with_sorting(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            order_by=order_by,
            sort_order=sort_order,
        )

        assert validate_items_sorted_correctly(response["items"], order_by, sort_order)

    @pytest.mark.parametrize("unsupported_field", ["CREATE_TIME", "LAST_UPDATE_TIME"])
    def test_sources_rejects_unsupported_fields(
        self: Self,
        enabled_model_catalog_config_map: ConfigMap,
        unsupported_field: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        RHOAIENG-37260: Test sources endpoint rejects fields it doesn't support
        """
        LOGGER.info(f"Testing sources rejection of unsupported field: {unsupported_field}")

        with pytest.raises(Exception, match="unsupported order by field"):
            get_sources_with_sorting(
                model_catalog_rest_url=model_catalog_rest_url,
                model_registry_rest_headers=model_registry_rest_headers,
                order_by=unsupported_field,
                sort_order="ASC",
            )


# More than 1 artifact are available only in downstream
@pytest.mark.downstream_only
class TestArtifactsSorting:
    """Test sorting functionality for GetAllModelArtifacts endpoint
    Fixed on a random model from the validated catalog since we need more than 1 artifact to test sorting.
    """

    @pytest.mark.parametrize(
        "order_by,sort_order,randomly_picked_model_from_catalog_api_by_source",
        [
            ("ID", "ASC", {"catalog_id": VALIDATED_CATALOG_ID, "header_type": "registry"}),
            ("ID", "DESC", {"catalog_id": VALIDATED_CATALOG_ID, "header_type": "registry"}),
            pytest.param(
                "NAME",
                "ASC",
                {"catalog_id": VALIDATED_CATALOG_ID, "header_type": "registry"},
                marks=pytest.mark.xfail(reason="RHOAIENG-38056: falls back to ID sorting"),
            ),
            pytest.param(
                "NAME",
                "DESC",
                {"catalog_id": VALIDATED_CATALOG_ID, "header_type": "registry"},
                marks=pytest.mark.xfail(reason="RHOAIENG-38056: falls back to ID sorting"),
            ),
        ],
        indirect=["randomly_picked_model_from_catalog_api_by_source"],
    )
    def test_artifacts_sorting_works_correctly(
        self: Self,
        enabled_model_catalog_config_map: ConfigMap,
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
@pytest.mark.smoke
class TestCustomPropertiesSorting:
    """Test sorting functionality for custom properties"""

    @pytest.mark.parametrize(
        "order_by,sort_order,randomly_picked_model_from_catalog_api_by_source,fallback",
        [
            ("e2e_p90.double_value", "ASC", {"catalog_id": VALIDATED_CATALOG_ID, "header_type": "registry"}, False),
            ("e2e_p90.double_value", "DESC", {"catalog_id": VALIDATED_CATALOG_ID, "header_type": "registry"}, False),
            ("mmlu.double_value", "ASC", {"catalog_id": VALIDATED_CATALOG_ID, "header_type": "registry"}, False),
            ("mmlu.double_value", "DESC", {"catalog_id": VALIDATED_CATALOG_ID, "header_type": "registry"}, False),
            (
                "non_existing_property.double_value",
                "ASC",
                {"catalog_id": VALIDATED_CATALOG_ID, "header_type": "registry"},
                True,
            ),
            (
                "non_existing_property.double_value",
                "DESC",
                {"catalog_id": VALIDATED_CATALOG_ID, "header_type": "registry"},
                True,
            ),
        ],
        indirect=["randomly_picked_model_from_catalog_api_by_source"],
    )
    def test_custom_properties_sorting_works_correctly(
        self: Self,
        enabled_model_catalog_config_map: ConfigMap,
        order_by: str,
        sort_order: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        randomly_picked_model_from_catalog_api_by_source: tuple[dict, str, str],
        fallback: bool,
    ):
        """
        RHOAIENG-38010: Test custom properties endpoint sorts correctly by supported fields
        Also tests fallback behavior when a non-existing property is used for sorting
        """
        _, model_name, _ = randomly_picked_model_from_catalog_api_by_source
        LOGGER.info(
            f"Testing custom properties sorting for {model_name}: "
            f"orderBy={order_by}, sortOrder={sort_order}, fallback={fallback}"
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

        if fallback:
            # Fallback test: verify NO artifacts have the property and sorting falls back to ID ASC
            # Note: Fallback always uses ASC order regardless of requested sortOrder
            assert artifacts_with_property == 0, (
                f"Expected no artifacts to have property {property_name} for fallback test, "
                f"but found {artifacts_with_property} artifacts with it"
            )
            is_sorted = validate_items_sorted_correctly(items=response["items"], field="ID", order="ASC")
            assert is_sorted, f"Fallback to ID ASC sorting failed for non-existing property {order_by}"
        else:
            # Normal test: verify at least some artifacts have the property
            assert artifacts_with_property > 0, (
                f"Cannot test custom property sorting: no artifacts have property {property_name}. "
                f"This would silently fall back to ID sorting."
            )
            LOGGER.info(f"{artifacts_with_property}/{len(response['items'])} artifacts have property {property_name}")

            is_sorted = verify_custom_properties_sorted(
                items=response["items"], property_field=order_by, sort_order=sort_order
            )
            assert is_sorted, f"Custom properties are not sorted correctly for {model_name}"
