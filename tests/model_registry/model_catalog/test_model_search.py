import pytest
from dictdiffer import diff

from simple_logger.logger import get_logger
from typing import Self, Any
from tests.model_registry.model_catalog.constants import (
    REDHATI_AI_FILTER,
    REDHATI_AI_VALIDATED_FILTER,
    REDHAT_AI_CATALOG_ID,
    VALIDATED_CATALOG_ID,
)
from tests.model_registry.model_catalog.utils import (
    execute_get_command,
)

LOGGER = get_logger(name=__name__)
pytestmark = [
    pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace", "test_idp_user")
]


class TestSearchModelCatalog:
    @pytest.mark.smoke
    def test_search_model_catalog_source_label(
        self: Self, model_catalog_rest_url: list[str], model_registry_rest_headers: str
    ):
        """
        RHOAIENG-33656: Validate search model catalog by source label
        """

        result = execute_get_command(
            url=f"{model_catalog_rest_url[0]}models?sourceLabel={REDHATI_AI_FILTER}&pageSize=100",
            headers=model_registry_rest_headers,
        )
        redhai_ai_filter_moldels_size = result["size"]

        result = execute_get_command(
            url=f"{model_catalog_rest_url[0]}models?sourceLabel={REDHATI_AI_VALIDATED_FILTER}&pageSize=100",
            headers=model_registry_rest_headers,
        )
        redhai_ai_validated_filter_models_size = result["size"]

        result = execute_get_command(
            url=f"{model_catalog_rest_url[0]}models?pageSize=100", headers=model_registry_rest_headers
        )
        no_filtered_models_size = result["size"]

        result = execute_get_command(
            url=(
                f"{model_catalog_rest_url[0]}models?"
                f"sourceLabel={REDHATI_AI_VALIDATED_FILTER},{REDHATI_AI_FILTER}&pageSize=100"
            ),
            headers=model_registry_rest_headers,
        )
        both_filtered_models_size = result["size"]

        assert no_filtered_models_size == both_filtered_models_size
        assert redhai_ai_filter_moldels_size + redhai_ai_validated_filter_models_size == both_filtered_models_size

    def test_search_model_catalog_invalid_source_label(
        self: Self, model_catalog_rest_url: list[str], model_registry_rest_headers: str
    ):
        """
        RHOAIENG-33656:
        Validate search model catalog by invalid source label
        """

        result = execute_get_command(
            url=f"{model_catalog_rest_url[0]}/models?sourceLabel=null&pageSize=100", headers=model_registry_rest_headers
        )
        null_size = result["size"]

        result = execute_get_command(
            url=f"{model_catalog_rest_url[0]}/models?sourceLabel=invalid&pageSize=100",
            headers=model_registry_rest_headers,
        )
        invalid_size = result["size"]

        assert null_size == invalid_size == 0, (
            "Expected 0 models for null and invalid source label found {null_size} and {invalid_size}"
        )

    @pytest.mark.parametrize(
        "randomly_picked_model,source_filter",
        [
            pytest.param(
                {"source": VALIDATED_CATALOG_ID},
                REDHATI_AI_VALIDATED_FILTER,
                id="test_search_model_catalog_redhat_ai_validated",
            ),
            pytest.param(
                {"source": REDHAT_AI_CATALOG_ID}, REDHATI_AI_FILTER, id="test_search_model_catalog_redhat_ai_default"
            ),
        ],
        indirect=["randomly_picked_model"],
    )
    def test_search_model_catalog_match(
        self: Self,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: str,
        randomly_picked_model: dict[Any, Any],
        source_filter: str,
    ):
        """
        RHOAIENG-33656: Validate search model catalog by match
        """
        random_model = randomly_picked_model
        random_model_name = random_model["name"]
        LOGGER.info(f"random_model_name: {random_model_name}")
        result = execute_get_command(
            url=(
                f"{model_catalog_rest_url[0]}/models?"
                f"sourceLabel={source_filter}&"
                f"filterQuery=name='{random_model_name}'&pageSize=100"
            ),
            headers=model_registry_rest_headers,
        )
        assert random_model_name == result["items"][0]["name"]
        assert result["size"] == 1

        differences = list(diff(random_model, result["items"][0]))
        assert not differences, f"Expected no differences in model information for {random_model_name}: {differences}"
        LOGGER.info("Model information matches")
