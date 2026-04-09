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
        "order_by, sort_order, only_huggingface_source",
        [
            pytest.param("ID", "ASC", True, id="test_hf_source_id_asc"),
            pytest.param("ID", "DESC", True, id="test_hf_source_id_desc"),
            pytest.param("NAME", "ASC", True, id="test_hf_source_name_asc"),
            pytest.param("NAME", "DESC", True, id="test_hf_source_name_desc"),
            pytest.param("CREATE_TIME", "ASC", True, id="test_hf_source_create_time_asc"),
            pytest.param("CREATE_TIME", "DESC", True, id="test_hf_source_create_time_desc"),
            pytest.param("LAST_UPDATE_TIME", "ASC", True, id="test_hf_source_last_update_time_asc"),
            pytest.param("LAST_UPDATE_TIME", "DESC", True, id="test_hf_source_last_update_time_desc"),
            pytest.param("ID", "ASC", False, id="test_all_sources_id_asc"),
            pytest.param("ID", "DESC", False, id="test_all_sources_id_desc"),
            pytest.param("NAME", "ASC", False, id="test_all_sources_name_asc"),
            pytest.param("NAME", "DESC", False, id="test_all_sources_name_desc"),
            pytest.param("CREATE_TIME", "ASC", False, id="test_all_sources_create_time_asc"),
            pytest.param("CREATE_TIME", "DESC", False, id="test_all_sources_create_time_desc"),
            pytest.param("LAST_UPDATE_TIME", "ASC", False, id="test_all_sources_last_update_time_asc"),
            pytest.param("LAST_UPDATE_TIME", "DESC", False, id="test_all_sources_last_update_time_desc"),
        ],
    )
    def test_huggingface_models_sorting_works_correctly(
        self: Self,
        order_by: str,
        sort_order: str,
        only_huggingface_source: bool,
        updated_catalog_config_map: ConfigMap,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        assert_model_sorting(
            order_by=order_by,
            sort_order=sort_order,
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_label="HuggingFace Source mixed" if only_huggingface_source else None,
        )
