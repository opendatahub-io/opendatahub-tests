import pytest
from typing import Self

from simple_logger.logger import get_logger
from tests.model_registry.model_catalog.utils import (
    get_models_from_catalog_api,
    get_sources_with_sorting,
    get_artifacts_with_sorting,
    validate_items_sorted_correctly,
)
from tests.model_registry.model_catalog.constants import VALIDATED_CATALOG_ID

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


# More than 1 artifact are available only in downstream
@pytest.mark.downstream_only
class TestArtifactsSorting:
    """Test sorting functionality for GetAllModelArtifacts endpoint
    Fixed on a random model from the validated catalog since we need more than 1 artifact to test sorting.
    """

    @pytest.mark.parametrize(
        "order_by,sort_order,randomly_picked_model_from_catalog_api_by_source",
        [
            ("ID", "ASC", {"catalog_id": VALIDATED_CATALOG_ID, "header_type": "registry"}),
            ("ID", "DESC", {"catalog_id": VALIDATED_CATALOG_ID, "header_type": "registry"}),
            pytest.param(
                "NAME",
                "ASC",
                {"catalog_id": VALIDATED_CATALOG_ID, "header_type": "registry"},
                marks=pytest.mark.xfail(reason="RHOAIENG-38056: falls back to ID sorting"),
            ),
            pytest.param(
                "NAME",
                "DESC",
                {"catalog_id": VALIDATED_CATALOG_ID, "header_type": "registry"},
                marks=pytest.mark.xfail(reason="RHOAIENG-38056: falls back to ID sorting"),
            ),
        ],
        indirect=["randomly_picked_model_from_catalog_api_by_source"],
    )
    def test_artifacts_sorting_works_correctly(
        self: Self,
        order_by: str,
        sort_order: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        randomly_picked_model_from_catalog_api_by_source: tuple[dict, str, str],
    ):
        """
        RHOAIENG-37260: Test artifacts endpoint sorts correctly by supported fields
        """
        _, model_name, _ = randomly_picked_model_from_catalog_api_by_source
        LOGGER.info(f"Testing artifacts sorting for {model_name}: orderBy={order_by}, sortOrder={sort_order}")

        response = get_artifacts_with_sorting(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_id=VALIDATED_CATALOG_ID,
            model_name=model_name,
            order_by=order_by,
            sort_order=sort_order,
        )

        assert validate_items_sorted_correctly(response["items"], order_by, sort_order)
