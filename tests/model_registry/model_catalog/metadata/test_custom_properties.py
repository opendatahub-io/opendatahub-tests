from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from simple_logger.logger import get_logger

from tests.model_registry.model_catalog.constants import REDHAT_AI_CATALOG_ID, VALIDATED_CATALOG_ID
from tests.model_registry.model_catalog.metadata.utils import (
    extract_custom_property_values,
    get_metadata_from_catalog_pod,
    validate_custom_properties_match_metadata,
)
from tests.model_registry.utils import execute_get_command, get_model_catalog_pod

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

    @pytest.mark.parametrize("catalog_id", [REDHAT_AI_CATALOG_ID, VALIDATED_CATALOG_ID])
    def test_model_type_field_in_custom_properties(
        self,
        catalog_id: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        RHOAIENG-41720: Test that all models have model_type with valid values: "generative", "predictive", "unknown".
        """
        valid_model_types = {"generative", "predictive", "unknown"}

        response = execute_get_command(
            url=f"{model_catalog_rest_url[0]}models?source={catalog_id}&pageSize=100",
            headers=model_registry_rest_headers,
        )
        models = response["items"]

        LOGGER.info(f"Validating model_type field for {len(models)} models from catalog '{catalog_id}'")

        validation_errors = []

        for model in models:
            custom_properties = model.get("customProperties", {})

            if "model_type" not in custom_properties:
                validation_errors.append(f"Model '{model.get('name')}' missing model_type in customProperties")
                continue

            model_type_value = custom_properties["model_type"]["string_value"]
            if model_type_value not in valid_model_types:
                validation_errors.append(
                    f"Model '{model.get('name')}' has invalid model_type: '{model_type_value}'. "
                    f"Expected one of: {valid_model_types}"
                )

        assert not validation_errors, (
            f"model_type validation failed for {len(validation_errors)} models:\n" + "\n".join(validation_errors)
        )

        LOGGER.info(f"All {len(models)} models in catalog '{catalog_id}' have valid model_type values")
