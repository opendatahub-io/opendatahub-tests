from typing import Self

import pytest
from simple_logger.logger import get_logger

from tests.model_registry.model_catalog.sorting.utils import (
    get_sources_with_sorting,
    validate_items_sorted_correctly,
)

LOGGER = get_logger(name=__name__)

pytestmark = [pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace")]


class TestSourcesSorting:
    """Test sorting functionality for FindSources endpoint"""

    @pytest.mark.parametrize(
        "order_by,sort_order",
        [
            ("ID", "ASC"),
            ("ID", "DESC"),
            ("NAME", "ASC"),
            ("NAME", "DESC"),
        ],
    )
    def test_sources_sorting_works_correctly(
        self: Self,
        order_by: str,
        sort_order: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        RHOAIENG-37260: Test sources endpoint sorts correctly by supported fields
        """
        LOGGER.info(f"Testing sources sorting: orderBy={order_by}, sortOrder={sort_order}")

        response = get_sources_with_sorting(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            order_by=order_by,
            sort_order=sort_order,
        )

        assert validate_items_sorted_correctly(response["items"], order_by, sort_order)

    @pytest.mark.parametrize("unsupported_field", ["CREATE_TIME", "LAST_UPDATE_TIME"])
    def test_sources_rejects_unsupported_fields(
        self: Self,
        unsupported_field: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        RHOAIENG-37260: Test sources endpoint rejects fields it doesn't support
        """
        LOGGER.info(f"Testing sources rejection of unsupported field: {unsupported_field}")

        with pytest.raises(Exception, match="unsupported order by field"):
            get_sources_with_sorting(
                model_catalog_rest_url=model_catalog_rest_url,
                model_registry_rest_headers=model_registry_rest_headers,
                order_by=unsupported_field,
                sort_order="ASC",
            )
