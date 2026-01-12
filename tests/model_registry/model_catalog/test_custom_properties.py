import pytest
from typing import Any

from kubernetes.dynamic import DynamicClient
from simple_logger.logger import get_logger

from tests.model_registry.model_catalog.constants import VALIDATED_CATALOG_ID
from tests.model_registry.model_catalog.utils import (
    extract_custom_property_values,
    validate_custom_properties_structure,
    validate_custom_properties_match_metadata,
    get_metadata_from_catalog_pod,
)
from tests.model_registry.utils import get_model_catalog_pod

LOGGER = get_logger(name=__name__)

pytestmark = [
    pytest.mark.usefixtures(
        "updated_dsc_component_state_scope_session",
        "model_registry_namespace",
    )
]


@pytest.mark.skip_must_gather
class TestCustomProperties:
    """Test suite for validating custom properties in model catalog API"""

    @pytest.mark.parametrize(
        "randomly_picked_model_from_catalog_api_by_source", [{"source": VALIDATED_CATALOG_ID}], indirect=True
    )
    def test_custom_properties_structure_is_valid(
        self,
        randomly_picked_model_from_catalog_api_by_source: tuple[dict[Any, Any], str, str],
    ):
        """Test that custom properties follow the expected MetadataStringValue structure."""
        model_data, model_name, catalog_id = randomly_picked_model_from_catalog_api_by_source

        LOGGER.info(f"Testing custom properties structure for model '{model_name}' from catalog '{catalog_id}'")

        custom_props = model_data.get("customProperties", {})
        assert validate_custom_properties_structure(custom_props)

    @pytest.mark.parametrize(
        "randomly_picked_model_from_catalog_api_by_source", [{"source": VALIDATED_CATALOG_ID}], indirect=True
    )
    def test_custom_properties_match_metadata(
        self,
        randomly_picked_model_from_catalog_api_by_source: tuple[dict[Any, Any], str, str],
        admin_client: DynamicClient,
        model_registry_namespace: str,
    ):
        """Test that custom properties from API match values in metadata.json files."""
        model_data, model_name, catalog_id = randomly_picked_model_from_catalog_api_by_source

        LOGGER.info(f"Testing custom properties metadata match for model '{model_name}' from catalog '{catalog_id}'")

        # Get model catalog pod
        model_catalog_pods = get_model_catalog_pod(
            client=admin_client, model_registry_namespace=model_registry_namespace
        )
        assert len(model_catalog_pods) > 0, "No model catalog pods found"

        # Extract custom properties and get metadata
        custom_props = model_data.get("customProperties", {})
        api_props = extract_custom_property_values(custom_properties=custom_props)
        metadata = get_metadata_from_catalog_pod(model_catalog_pod=model_catalog_pods[0], model_name=model_name)

        assert validate_custom_properties_match_metadata(api_props, metadata)
