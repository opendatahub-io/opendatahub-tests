import pytest
from simple_logger.logger import get_logger

from tests.model_registry.model_catalog.constants import REDHAT_AI_CATALOG_ID
from tests.model_registry.model_catalog.utils import execute_get_command

LOGGER = get_logger(name=__name__)

pytestmark = [pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace")]


class TestCatalogSourceMerge:
    """
    Test catalog source merging behavior when the same source ID appears in both
    default and custom ConfigMaps.
    """

    @pytest.mark.parametrize(
        "sparse_override_catalog_source",
        [
            {"id": REDHAT_AI_CATALOG_ID, "field_name": "name", "field_value": "Custom Override Name"},
            {"id": REDHAT_AI_CATALOG_ID, "field_name": "labels", "field_value": ["custom-label", "override-label"]},
            {"id": REDHAT_AI_CATALOG_ID, "field_name": "enabled", "field_value": False},
        ],
        indirect=True,
    )
    def test_catalog_source_merge(
        self,
        sparse_override_catalog_source: dict,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        RHOAIENG-41738: Test that a sparse override in custom ConfigMap successfully overrides
        specific fields while preserving unspecified fields.
        """
        catalog_id = sparse_override_catalog_source["catalog_id"]
        field_name = sparse_override_catalog_source["field_name"]
        field_value = sparse_override_catalog_source["field_value"]
        original_catalog = sparse_override_catalog_source["original_catalog"]

        # Query sources endpoint to get the merged result
        response = execute_get_command(url=f"{model_catalog_rest_url[0]}sources", headers=model_registry_rest_headers)
        items = response.get("items", [])
        LOGGER.info(f"API response items: {items}")

        # Find the catalog we're testing
        merged_catalog = next((item for item in items if item.get("id") == catalog_id), None)
        assert merged_catalog is not None, f"Catalog '{catalog_id}' not found in sources"

        # Validate the overridden field has the new value
        assert merged_catalog.get(field_name) == field_value, (
            f"Field '{field_name}' should have value {field_value}, got: {merged_catalog.get(field_name)}"
        )

        # Determine fields to check for preservation (exclude the overridden field)
        fields_to_check = set(original_catalog.keys()) - {field_name}

        # Special case: when enabled=False, status automatically changes to "disabled"
        if field_name == "enabled" and field_value is False:
            assert merged_catalog.get("status") == "disabled", (
                f"Status should be 'disabled' when enabled=False, got: {merged_catalog.get('status')}"
            )
            fields_to_check.discard("status")

        # Validate all other fields preserve original values
        for orig_field in fields_to_check:
            orig_value = original_catalog[orig_field]
            assert merged_catalog.get(orig_field) == orig_value, (
                f"Field '{orig_field}' should preserve original value {orig_value}, "
                f"got: {merged_catalog.get(orig_field)}"
            )

        LOGGER.info(
            f"Sparse override merge validated for '{catalog_id}' - "
            f"Overridden field '{field_name}': {field_value} | "
            f"Preserved {len(original_catalog) - 1} other fields"
        )
