from typing import Self

import pytest
from ocp_resources.config_map import ConfigMap

from tests.model_registry.model_catalog.sorting.utils import assert_model_sorting
from tests.model_registry.model_catalog.utils import get_hf_catalog_str

pytestmark = [pytest.mark.skip_on_disconnected]


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
            ("NAME", "ASC"),
            ("NAME", "DESC"),
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
