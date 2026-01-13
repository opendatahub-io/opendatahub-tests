from typing import Self

import pytest

from ocp_resources.config_map import ConfigMap
from tests.model_registry.model_catalog.utils import get_hf_catalog_str, get_models_from_catalog_api
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)


@pytest.mark.parametrize(
    "updated_catalog_config_map, hf_model_name, source_filter",
    [
        pytest.param(
            {
                "sources_yaml": get_hf_catalog_str(ids=["mixed"]),
            },
            "ibm-granite/granite-4.0-h-1b",
            "HuggingFace Source mixed",
            id="test_huggingface_model_filter_by_name",
        ),
    ],
    indirect=["updated_catalog_config_map"],
)
class TestHuggingFaceModelSearch:
    def test_search_model_catalog_huggingface(
        self: Self,
        updated_catalog_config_map: ConfigMap,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        hf_model_name: str,
        source_filter: str,
    ):
        """
        RHOAIENG-41869: Validate search model catalog by match
        """
        LOGGER.info(f"Testing ability to filter models by name: {hf_model_name}")
        result = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_label=source_filter,
            additional_params=f"&filterQuery=name='{hf_model_name}'",
        )
        LOGGER.info(result)
        assert hf_model_name == result["items"][0]["name"]
        assert result["size"] == 1
        LOGGER.info("Model information matches")
