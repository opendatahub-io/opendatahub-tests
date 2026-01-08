import pytest

from ocp_resources.config_map import ConfigMap
from simple_logger.logger import get_logger

from tests.model_registry.utils import execute_get_command
from tests.model_registry.model_catalog.metadata.utils import validate_source_status, validate_source_error

pytestmark = [pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace")]

LOGGER = get_logger(name=__name__)


class TestSourcesEndpoint:
    """Test class for the model catalog sources endpoint status and error fields, RHOAIENG-41849."""

    @pytest.mark.smoke
    def test_available_source_status(
        self,
        enabled_model_catalog_config_map: ConfigMap,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        Test that the sources endpoint returns no error for available sources.
        """
        response = execute_get_command(url=f"{model_catalog_rest_url[0]}sources", headers=model_registry_rest_headers)
        items = response.get("items", [])
        assert items, "Sources not found"
        for item in items:
            validate_source_status(catalog=item, expected_status="available")
            validate_source_error(catalog=item)
            LOGGER.info(
                "Available catalog verified - "
                f"ID: {item.get('id')}, "
                f"Status: {item.get('status')}, "
                f"Error: {item.get('error')}"
            )

    @pytest.mark.parametrize("disabled_catalog_source", ["redhat_ai_models"], indirect=True)
    def test_disabled_source_status(
        self,
        enabled_model_catalog_config_map: ConfigMap,
        disabled_catalog_source: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        This test disables an existing catalog and verifies:
        - status field is "disabled"
        - error field is null or empty
        """
        catalog_id = disabled_catalog_source

        response = execute_get_command(url=f"{model_catalog_rest_url[0]}sources", headers=model_registry_rest_headers)
        items = response.get("items", [])

        # Find the disabled catalog
        disabled_catalog = next((item for item in items if item.get("id") == catalog_id), None)
        assert disabled_catalog is not None, f"Disabled catalog '{catalog_id}' not found in sources"

        # Validate status and error fields
        validate_source_status(catalog=disabled_catalog, expected_status="disabled")
        validate_source_error(catalog=disabled_catalog)

        LOGGER.info(
            "Disabled catalog verified - "
            f"ID: {disabled_catalog.get('id')}, "
            f"Status: {disabled_catalog.get('status')}, "
            f"Error: {disabled_catalog.get('error')}"
        )
