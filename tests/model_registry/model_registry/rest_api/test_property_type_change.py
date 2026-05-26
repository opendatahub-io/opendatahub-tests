from typing import Any, Self

import pytest
import structlog

from tests.model_registry.model_registry.rest_api.constants import MODEL_REGISTRY_BASE_URI
from tests.model_registry.model_registry.rest_api.utils import (
    execute_model_registry_patch_command,
    execute_model_registry_post_command,
)
from tests.model_registry.utils import execute_model_registry_get_command

LOGGER = structlog.get_logger(name=__name__)

PROPERTY_NAME: str = "score"


@pytest.mark.parametrize(
    "model_registry_metadata_db_resources, model_registry_instance",
    [
        pytest.param(
            {"db_name": "postgres"},
            {"db_name": "postgres"},
            marks=pytest.mark.tier1,
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures(
    "updated_dsc_component_state_scope_session",
    "model_registry_namespace",
    "model_registry_metadata_db_resources",
    "model_registry_instance",
)
@pytest.mark.custom_namespace
class TestPropertyTypeChange:
    """Regression tests for RHOAIENG-59192.

    Validates that changing a custom property's value type clears the
    previous type's column in the database, preventing stale data.
    """

    @pytest.fixture(scope="class")
    def registered_model_with_int_property(
        self: Self,
        model_registry_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ) -> dict[str, Any]:
        """Registered model with an integer custom property."""
        return execute_model_registry_post_command(
            url=f"{model_registry_rest_url[0]}{MODEL_REGISTRY_BASE_URI}registered_models",
            headers=model_registry_rest_headers,
            data_json={
                "name": "repro-stale-prop-test",
                "customProperties": {
                    PROPERTY_NAME: {
                        "metadataType": "MetadataIntValue",
                        "int_value": "42",
                    }
                },
            },
        )

    @pytest.mark.dependency(name="create_model")
    def test_initial_int_property(
        self: Self,
        model_registry_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        registered_model_with_int_property: dict[str, Any],
    ) -> None:
        """Given a newly created model with an integer custom property,
        when the model is retrieved via the REST API,
        then the property should contain only the integer value.
        """
        model_id = registered_model_with_int_property["id"]
        model = execute_model_registry_get_command(
            url=f"{model_registry_rest_url[0]}{MODEL_REGISTRY_BASE_URI}registered_models/{model_id}",
            headers=model_registry_rest_headers,
        )
        prop = model["customProperties"][PROPERTY_NAME]
        assert prop["metadataType"] == "MetadataIntValue"
        assert prop["int_value"] == "42"

    @pytest.mark.dependency(name="int_to_double", depends=["create_model"])
    def test_int_to_double_clears_stale_int(
        self: Self,
        model_registry_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        registered_model_with_int_property: dict[str, Any],
    ) -> None:
        """Given a model whose custom property was originally an integer,
        when the property type is changed to double via PATCH,
        then the API response should contain only double_value and no
        residual int_value key.
        """
        model_id = registered_model_with_int_property["id"]
        execute_model_registry_patch_command(
            url=f"{model_registry_rest_url[0]}{MODEL_REGISTRY_BASE_URI}registered_models/{model_id}",
            headers=model_registry_rest_headers,
            data_json={
                "customProperties": {
                    PROPERTY_NAME: {
                        "metadataType": "MetadataDoubleValue",
                        "double_value": 3.14,
                    }
                }
            },
        )
        model = execute_model_registry_get_command(
            url=f"{model_registry_rest_url[0]}{MODEL_REGISTRY_BASE_URI}registered_models/{model_id}",
            headers=model_registry_rest_headers,
        )
        prop = model["customProperties"][PROPERTY_NAME]
        assert prop["metadataType"] == "MetadataDoubleValue", (
            f"Expected MetadataDoubleValue, got {prop['metadataType']}"
        )
        assert prop["double_value"] == 3.14
        assert "int_value" not in prop, f"Stale int_value found in API response: {prop}"

    @pytest.mark.dependency(name="double_to_string", depends=["int_to_double"])
    def test_double_to_string_clears_stale_double(
        self: Self,
        model_registry_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        registered_model_with_int_property: dict[str, Any],
    ) -> None:
        """Given a model whose custom property was changed to double,
        when the property type is changed to string via PATCH,
        then the API response should contain only string_value and no
        residual double_value or int_value keys.
        """
        model_id = registered_model_with_int_property["id"]
        execute_model_registry_patch_command(
            url=f"{model_registry_rest_url[0]}{MODEL_REGISTRY_BASE_URI}registered_models/{model_id}",
            headers=model_registry_rest_headers,
            data_json={
                "customProperties": {
                    PROPERTY_NAME: {
                        "metadataType": "MetadataStringValue",
                        "string_value": "high",
                    }
                }
            },
        )
        model = execute_model_registry_get_command(
            url=f"{model_registry_rest_url[0]}{MODEL_REGISTRY_BASE_URI}registered_models/{model_id}",
            headers=model_registry_rest_headers,
        )
        prop = model["customProperties"][PROPERTY_NAME]
        assert prop["metadataType"] == "MetadataStringValue", (
            f"Expected MetadataStringValue, got {prop['metadataType']}"
        )
        assert prop["string_value"] == "high"
        assert "int_value" not in prop, f"Stale int_value found in API response: {prop}"
        assert "double_value" not in prop, f"Stale double_value found in API response: {prop}"

    @pytest.mark.dependency(name="string_to_int", depends=["double_to_string"])
    def test_string_to_int_clears_stale_string(
        self: Self,
        model_registry_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        registered_model_with_int_property: dict[str, Any],
    ) -> None:
        """Given a model whose custom property was changed to string,
        when the property type is changed back to integer via PATCH,
        then the API response should contain only int_value and no
        residual string_value or double_value keys.
        """
        model_id = registered_model_with_int_property["id"]
        execute_model_registry_patch_command(
            url=f"{model_registry_rest_url[0]}{MODEL_REGISTRY_BASE_URI}registered_models/{model_id}",
            headers=model_registry_rest_headers,
            data_json={
                "customProperties": {
                    PROPERTY_NAME: {
                        "metadataType": "MetadataIntValue",
                        "int_value": "99",
                    }
                }
            },
        )
        model = execute_model_registry_get_command(
            url=f"{model_registry_rest_url[0]}{MODEL_REGISTRY_BASE_URI}registered_models/{model_id}",
            headers=model_registry_rest_headers,
        )
        prop = model["customProperties"][PROPERTY_NAME]
        assert prop["metadataType"] == "MetadataIntValue", f"Expected MetadataIntValue, got {prop['metadataType']}"
        assert prop["int_value"] == "99"
        assert "string_value" not in prop, f"Stale string_value found in API response: {prop}"
        assert "double_value" not in prop, f"Stale double_value found in API response: {prop}"
