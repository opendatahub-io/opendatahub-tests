import pytest
from simple_logger.logger import get_logger

from tests.model_registry.utils import execute_get_command

LOGGER = get_logger(name=__name__)

pytestmark = [pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace")]


class TestCatalogSourceMerge:
    """
    Test catalog source merging behavior when the same source ID appears in both
    default and custom ConfigMaps.
    """

    def test_catalog_source_merge(
        self,
        sparse_override_catalog_source: dict,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        RHOAIENG-41738:Test that a sparse override in custom ConfigMap successfully overrides
        specific fields while preserving unspecified fields.

        Expected behavior:
        - name and labels should have new values
        - All other fields (enabled, status, error) should preserve original values
        """
        catalog_id = sparse_override_catalog_source["catalog_id"]
        custom_name = sparse_override_catalog_source["custom_name"]
        custom_labels = sparse_override_catalog_source["custom_labels"]
        original_catalog = sparse_override_catalog_source["original_catalog"]

        # Query sources endpoint to get the merged result
        response = execute_get_command(url=f"{model_catalog_rest_url[0]}sources", headers=model_registry_rest_headers)
        items = response.get("items", [])
        LOGGER.info(f"API response items: {items}")

        # Find the catalog we're testing
        merged_catalog = next((item for item in items if item.get("id") == catalog_id), None)
        assert merged_catalog is not None, f"Catalog '{catalog_id}' not found in sources"

        # Define expected values: overridden fields get new values, others preserve original
        expected_values = {
            "name": custom_name,
            "labels": custom_labels,
        }

        validation_errors = []
        for field_name, original_value in original_catalog.items():
            # Skip id (used for catalog matching)
            if field_name == "id":
                continue

            merged_value = merged_catalog.get(field_name)

            # Check if this field should be overridden or preserved
            if field_name in expected_values:
                # This field was overridden in sparse catalog
                expected_value = expected_values[field_name]
                if merged_value != expected_value:
                    validation_errors.append(
                        f"Field '{field_name}' should be overridden to {expected_value}, got: {merged_value}"
                    )
            elif merged_value != original_value:
                validation_errors.append(
                    f"Field '{field_name}' should preserve original value {original_value}, got: {merged_value}"
                )

        assert not validation_errors, "Some fields did not have expected values:\n" + "\n".join(validation_errors)

        LOGGER.info(
            f"Sparse override merge validated for '{catalog_id}' - "
            f"Overridden fields: {list(expected_values.keys())} | "
            f"Preserved fields: {[f for f in original_catalog.keys() if f not in expected_values and f != 'id']}"
        )
