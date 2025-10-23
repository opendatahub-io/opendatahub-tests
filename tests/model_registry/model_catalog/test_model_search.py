import pytest
from dictdiffer import diff

from simple_logger.logger import get_logger
from typing import Self, Any
from tests.model_registry.model_catalog.constants import (
    REDHAT_AI_FILTER,
    REDHAT_AI_VALIDATED_FILTER,
    REDHAT_AI_CATALOG_ID,
    VALIDATED_CATALOG_ID,
)
from tests.model_registry.model_catalog.utils import (
    get_models_from_api,
)

LOGGER = get_logger(name=__name__)
pytestmark = [
    pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace", "test_idp_user")
]


class TestSearchModelCatalog:
    @pytest.mark.smoke
    def test_search_model_catalog_source_label(
        self: Self, model_catalog_rest_url: list[str], model_registry_rest_headers: dict[str, str]
    ):
        """
        RHOAIENG-33656: Validate search model catalog by source label
        """

        redhat_ai_filter_moldels_size = get_models_from_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_label=REDHAT_AI_FILTER,
        )["size"]
        redhat_ai_validated_filter_models_size = get_models_from_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_label=REDHAT_AI_VALIDATED_FILTER,
        )["size"]
        no_filtered_models_size = get_models_from_api(
            model_catalog_rest_url=model_catalog_rest_url, model_registry_rest_headers=model_registry_rest_headers
        )["size"]
        both_filtered_models_size = get_models_from_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_label=f"{REDHAT_AI_VALIDATED_FILTER},{REDHAT_AI_FILTER}",
        )["size"]

        assert no_filtered_models_size == both_filtered_models_size
        assert redhat_ai_filter_moldels_size + redhat_ai_validated_filter_models_size == both_filtered_models_size

    def test_search_model_catalog_invalid_source_label(
        self: Self, model_catalog_rest_url: list[str], model_registry_rest_headers: dict[str, str]
    ):
        """
        RHOAIENG-33656:
        Validate search model catalog by invalid source label
        """

        null_size = get_models_from_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_label="null",
        )["size"]

        invalid_size = get_models_from_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_label="invalid",
        )["size"]

        assert null_size == invalid_size == 0, (
            "Expected 0 models for null and invalid source label found {null_size} and {invalid_size}"
        )

    @pytest.mark.parametrize(
        "randomly_picked_model,source_filter",
        [
            pytest.param(
                {"source": VALIDATED_CATALOG_ID},
                REDHAT_AI_VALIDATED_FILTER,
                id="test_search_model_catalog_redhat_ai_validated",
            ),
            pytest.param(
                {"source": REDHAT_AI_CATALOG_ID}, REDHAT_AI_FILTER, id="test_search_model_catalog_redhat_ai_default"
            ),
        ],
        indirect=["randomly_picked_model"],
    )
    def test_search_model_catalog_match(
        self: Self,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        randomly_picked_model: dict[Any, Any],
        source_filter: str,
    ):
        """
        RHOAIENG-33656: Validate search model catalog by match
        """
        random_model = randomly_picked_model
        random_model_name = random_model["name"]
        LOGGER.info(f"random_model_name: {random_model_name}")
        result = get_models_from_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_label=source_filter,
            additional_params=f"&filterQuery=name='{random_model_name}'",
        )
        assert random_model_name == result["items"][0]["name"]
        assert result["size"] == 1

        differences = list(diff(random_model, result["items"][0]))
        assert not differences, f"Expected no differences in model information for {random_model_name}: {differences}"
        LOGGER.info("Model information matches")
