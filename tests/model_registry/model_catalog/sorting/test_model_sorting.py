import pytest
from typing import Self
from ocp_resources.config_map import ConfigMap
from simple_logger.logger import get_logger
from tests.model_registry.model_catalog.utils import get_models_from_catalog_api, get_hf_catalog_str
from tests.model_registry.model_catalog.sorting.utils import (
    validate_accuracy_sorting_against_database,
    assert_model_sorting,
)

LOGGER = get_logger(name=__name__)

pytestmark = [pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace")]


class TestModelsSorting:
    """Test sorting functionality for FindModels endpoint"""

    @pytest.mark.parametrize(
        "order_by,sort_order",
        [
            ("ID", "ASC"),
            ("ID", "DESC"),
            ("NAME", "ASC"),
            ("NAME", "DESC"),
            ("CREATE_TIME", "ASC"),
            ("CREATE_TIME", "DESC"),
            ("LAST_UPDATE_TIME", "ASC"),
            ("LAST_UPDATE_TIME", "DESC"),
        ],
    )
    def test_models_sorting_works_correctly(
        self: Self,
        order_by: str,
        sort_order: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        RHOAIENG-37260: Test models endpoint sorts correctly by field and order
        """
        assert_model_sorting(
            order_by=order_by,
            sort_order=sort_order,
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
        )


@pytest.mark.downstream_only
class TestAccuracySorting:
    """Test sorting for accuracy value in FindModels endpoint"""

    @pytest.mark.parametrize(
        "sort_order",
        [
            None,  # orderBy=ACCURACY without sortOrder
            "ASC",
            "DESC",
        ],
    )
    def test_accuracy_sorting_works_correctly(
        self: Self,
        sort_order: str | None,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        RHOAIENG-36856: Test accuracy sorting for FindModels endpoint

        This test validates accuracy sorting behavior with different sort_order parameters:

        When sort_order is None (orderBy=ACCURACY only):
        1. Models WITH accuracy appear first (in any order)
        2. Models WITHOUT accuracy appear after, sorted by ID in ASC order

        When sort_order is ASC or DESC (orderBy=ACCURACY&sortOrder=ASC/DESC):
        1. Models WITH accuracy appear first, sorted by accuracy value (ASC/DESC)
        2. Models WITHOUT accuracy appear after, sorted by ID in ASC order

        Validates both the presence of models and their correct ordering by comparing
        against direct database queries.
        """
        LOGGER.info(f"Testing accuracy sorting: sortOrder={sort_order}")

        response = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            order_by="ACCURACY",
            sort_order=sort_order,
        )

        assert validate_accuracy_sorting_against_database(
            api_response=response,
            sort_order=sort_order,
        )

    @pytest.mark.parametrize(
        "sort_order,filter_query",
        [
            ("ASC", "tasks='automatic-speech-translation'"),  # No models with accuracy
            ("ASC", "tasks='image-text-to-text'"),
            ("DESC", "tasks='image-text-to-text'"),
        ],
    )
    def test_accuracy_sorting_works_correctly_with_filter(
        self: Self,
        sort_order: str,
        filter_query: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        RHOAIENG-36856: Test accuracy sorting for FindModels endpoint with filter

        This test validates accuracy sorting behavior with task filter:
        1. Models WITH accuracy (and matching filter) appear first, sorted by accuracy value
        2. Models WITHOUT accuracy (but matching filter) appear after, sorted by ID in ASC order

        Validates both the presence of models and their correct ordering by comparing
        against direct database queries.
        """
        LOGGER.info(f"Testing accuracy sorting with filter: sortOrder={sort_order}, filterQuery={filter_query}")

        response = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            order_by="ACCURACY",
            sort_order=sort_order,
            additional_params=f"&filterQuery={filter_query}",
        )

        task_value = filter_query.split("tasks=")[1].strip("'\"")

        assert validate_accuracy_sorting_against_database(
            api_response=response,
            sort_order=sort_order,
            task_filter=task_value,
        )


@pytest.mark.parametrize(
    "updated_catalog_config_map",
    [
        pytest.param(
            {
                "sources_yaml": get_hf_catalog_str(ids=["mixed"]),
            },
            id="test_huggingface_model_sorting",
            marks=(pytest.mark.install),
        ),
    ],
    indirect=True,
)
class TestHuggingFaceModelsSorting:
    @pytest.mark.parametrize(
        "order_by,sort_order",
        [
            ("ID", "ASC"),
            ("ID", "DESC"),
            pytest.param(
                "NAME",
                "ASC",
            ),
            pytest.param(
                "NAME",
                "DESC",
            ),
            ("CREATE_TIME", "ASC"),
            ("CREATE_TIME", "DESC"),
            ("LAST_UPDATE_TIME", "ASC"),
            ("LAST_UPDATE_TIME", "DESC"),
        ],
    )
    def test_huggingface_models_sorting_works_correctly(
        self: Self,
        order_by: str,
        sort_order: str,
        updated_catalog_config_map: ConfigMap,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        assert_model_sorting(
            order_by=order_by,
            sort_order=sort_order,
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
        )
