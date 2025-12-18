import pytest
from typing import Self
from ocp_resources.config_map import ConfigMap
from simple_logger.logger import get_logger

from tests.model_registry.model_catalog.constants import HF_MODELS
from tests.model_registry.model_catalog.utils import (
    get_hf_catalog_str,
)
from tests.model_registry.model_catalog.huggingface.utils import (
    assert_huggingface_values_matches_model_catalog_api_values,
)

LOGGER = get_logger(name=__name__)

pytestmark = [pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace")]


@pytest.mark.parametrize(
    "updated_catalog_config_map, expected_catalog_values",
    [
        pytest.param(
            {
                "sources_yaml": get_hf_catalog_str(ids=["mixed"]),
            },
            HF_MODELS["mixed"],
            id="validate_hf_fields",
            marks=pytest.mark.install,
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("updated_catalog_config_map")
class TestHuggingFaceModelValidation:
    """Test HuggingFace model values by comparing values between HF API calls and Model Catalog api call"""

    def test_huggingface_model_metadata(
        self: Self,
        updated_catalog_config_map: tuple[ConfigMap, str, str],
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        expected_catalog_values: dict[str, str],
        huggingface_api: bool,
    ):
        """
        Validate HuggingFace model metadata structure and required fields
        Cross-validate with actual HuggingFace Hub API
        """
        assert_huggingface_values_matches_model_catalog_api_values(
            model_registry_rest_headers=model_registry_rest_headers,
            model_catalog_rest_url=model_catalog_rest_url,
            expected_catalog_values=expected_catalog_values,
            huggingface_api=huggingface_api,
        )
