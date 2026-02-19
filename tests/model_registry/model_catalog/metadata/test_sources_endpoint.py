import pytest

from simple_logger.logger import get_logger

from tests.model_registry.model_catalog.constants import REDHAT_AI_CATALOG_ID
from tests.model_registry.utils import execute_get_command

pytestmark = [pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace")]

LOGGER = get_logger(name=__name__)


class TestSourcesEndpoint:
    """Test class for the model catalog sources endpoint."""

    @pytest.mark.parametrize(
        "sparse_override_catalog_source",
        [{"id": REDHAT_AI_CATALOG_ID, "field_name": "enabled", "field_value": False}],
        indirect=True,
    )
    @pytest.mark.sanity
    def test_sources_endpoint_returns_all_sources_regardless_of_enabled_field(
        self,
        sparse_override_catalog_source: dict,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        RHOAIENG-41633: Test that sources endpoint returns ALL sources regardless of enabled field value.
        """
        response = execute_get_command(url=f"{model_catalog_rest_url[0]}sources", headers=model_registry_rest_headers)
        items = response.get("items", [])

        assert len(items) > 1, "Expected multiple sources to be returned"

        # Verify we have at least one enabled source
        enabled_sources = [item for item in items if item.get("status") == "available"]
        assert enabled_sources, "Expected at least one enabled source"

        # Verify we have at least one disabled source
        disabled_sources = [item for item in items if item.get("status") == "disabled"]
        assert disabled_sources, "Expected at least one disabled source"

        assert len(enabled_sources) + len(disabled_sources) == len(items), "Expected all sources to be returned"

        LOGGER.info(
            f"Sources endpoint returned {len(items)} total sources: "
            f"{len(enabled_sources)} enabled, {len(disabled_sources)} disabled"
        )
