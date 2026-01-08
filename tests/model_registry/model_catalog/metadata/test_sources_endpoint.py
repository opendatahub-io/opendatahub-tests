import pytest

from ocp_resources.config_map import ConfigMap
from simple_logger.logger import get_logger

from tests.model_registry.utils import execute_get_command
from tests.model_registry.model_catalog.metadata.utils import validate_source_status

pytestmark = [pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace")]

LOGGER = get_logger(name=__name__)


class TestSourcesEndpoint:
    """Test class for the model catalog sources endpoint."""

    @pytest.mark.parametrize("disabled_catalog_source", ["redhat_ai_models"], indirect=True)
    @pytest.mark.sanity
    def test_sources_endpoint_status_and_error_fields(
        self,
        enabled_model_catalog_config_map: ConfigMap,
        disabled_catalog_source: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        RHOAIENG-41243: Sources endpoint should return both enabled and disabled sources.
        RHOAIENG-41849: Sources should have status and error fields with correct values.
        """
        catalog_id = disabled_catalog_source

        response = execute_get_command(url=f"{model_catalog_rest_url[0]}sources", headers=model_registry_rest_headers)
        items = response.get("items", [])

        # Verify response contains multiple sources
        assert items, "Expected sources to be returned"

        # Split sources by status
        enabled_sources = [item for item in items if item.get("status") == "available"]
        disabled_sources = [item for item in items if item.get("status") == "disabled"]

        # Verify we have both enabled and disabled sources
        assert enabled_sources, "Expected at least one enabled source"
        assert disabled_sources, "Expected at least one disabled source"
        assert len(enabled_sources) + len(disabled_sources) == len(items), (
            "All sources should have either 'available' or 'disabled' status"
        )

        # Validate all enabled sources have correct status and no error
        for source in enabled_sources:
            validate_source_status(catalog=source, expected_status="available")
            error_value = source["error"]
            assert error_value is None or error_value == "", (
                f"Enabled source '{source.get('id')}' should not have error, got: {error_value}"
            )

        # Find and validate the specific disabled catalog from fixture
        disabled_catalog = next((item for item in disabled_sources if item.get("id") == catalog_id), None)
        assert disabled_catalog is not None, f"Disabled catalog '{catalog_id}' not found in sources"

        # Validate status and error fields for disabled catalog
        validate_source_status(catalog=disabled_catalog, expected_status="disabled")
        error_value = disabled_catalog["error"]
        assert error_value is None or error_value == "", (
            f"Disabled source '{disabled_catalog.get('id')}' should not have error, got: {error_value}"
        )

        LOGGER.info(
            f"Sources endpoint validation complete - Total: {len(items)}, "
            f"Enabled: {len(enabled_sources)}, Disabled: {len(disabled_sources)}"
        )
