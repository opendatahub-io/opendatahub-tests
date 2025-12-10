import pytest
from typing import Self
from ocp_resources.config_map import ConfigMap
from simple_logger.logger import get_logger
from tests.model_registry.model_catalog.utils import get_models_from_catalog_api
from tests.model_registry.model_catalog.sorting.utils import validate_items_sorted_correctly

LOGGER = get_logger(name=__name__)

pytestmark = [pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace")]


class TestModelsSorting:
    """Test sorting functionality for FindModels endpoint"""

    @pytest.mark.parametrize(
        "order_by,sort_order",
        [
            ("ID", "ASC"),
            ("ID", "DESC"),
            pytest.param(
                "NAME",
                "ASC",
                marks=pytest.mark.xfail(
                    reason="RHOAIENG-38056: Backend bug - NAME sorting not implemented, falls back to ID sorting"
                ),
            ),
            pytest.param(
                "NAME",
                "DESC",
                marks=pytest.mark.xfail(
                    reason="RHOAIENG-38056: Backend bug - NAME sorting not implemented, falls back to ID sorting"
                ),
            ),
            ("CREATE_TIME", "ASC"),
            ("CREATE_TIME", "DESC"),
            ("LAST_UPDATE_TIME", "ASC"),
            ("LAST_UPDATE_TIME", "DESC"),
        ],
    )
    def test_models_sorting_works_correctly(
        self: Self,
        enabled_model_catalog_config_map: ConfigMap,
        order_by: str,
        sort_order: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        RHOAIENG-37260: Test models endpoint sorts correctly by field and order
        """
        LOGGER.info(f"Testing models sorting: orderBy={order_by}, sortOrder={sort_order}")

        response = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            order_by=order_by,
            sort_order=sort_order,
        )

        assert validate_items_sorted_correctly(response["items"], order_by, sort_order)
