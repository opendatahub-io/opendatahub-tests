import pytest
from typing import Self
from ocp_resources.config_map import ConfigMap
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutExpiredError
from tests.model_registry.model_catalog.constants import HF_MODELS
from tests.model_registry.model_catalog.utils import (
    get_hf_catalog_str,
)
from tests.model_registry.model_catalog.huggingface.utils import (
    assert_huggingface_values_matches_model_catalog_api_values,
    wait_for_huggingface_retrival_match,
    get_model_catalog_pod,
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


class TestHFPatternMatching:
    @pytest.mark.parametrize(
        "updated_catalog_config_map_scope_function, expected_num_models_from_hf_api",
        [
            pytest.param(
                """
catalogs:
  - name: HuggingFace Hub
    id: hf_id
    type: hf
    enabled: true
    includedModels:
    - ibm-granite/*
""",
                {"org_name": "ibm-granite", "excluded_str": None},
                id="test_hf_source_wildcard",
            ),
            pytest.param(
                """
catalogs:
  - name: HuggingFace Hub
    id: hf_id
    type: hf
    enabled: true
    properties:
      allowedOrganization: "ibm-granite"
    includedModels:
    - '*'
""",
                {"org_name": "ibm-granite", "excluded_str": None},
                id="test_hf_source_allowed_org",
            ),
            pytest.param(
                """
catalogs:
  - name: HuggingFace Hub
    id: hf_id
    type: hf
    enabled: true
    properties:
      allowedOrganization: "ibm-granite"
    includedModels:
    - '*'
    excludedModels:
    - '*-base'
""",
                {"org_name": "ibm-granite", "excluded_str": "-base"},
                id="test_hf_source_allowed_org_exclude",
            ),
            pytest.param(
                """
catalogs:
  - name: HuggingFace Hub
    id: hf_id
    type: hf
    enabled: true
    includedModels:
    - 'ibm-granite/granite-4.0-micro*'
""",
                {"org_name": "ibm-granite", "included_str": "ibm-granite/granite-4.0-micro"},
                id="test_hf_source_allowed_org_include",
            ),
        ],
        indirect=True,
    )
    def test_hugging_face_models(
        self: Self,
        updated_catalog_config_map_scope_function: ConfigMap,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        huggingface_api: bool,
        expected_num_models_from_hf_api: int,
    ):
        """
        Test that excluded models do not appear in the catalog API response
        """
        LOGGER.info("Testing HuggingFace model exclusion functionality")
        try:
            wait_for_huggingface_retrival_match(
                model_registry_rest_headers=model_registry_rest_headers,
                model_catalog_rest_url=model_catalog_rest_url,
                expected_num_models_from_hf_api=expected_num_models_from_hf_api,
            )
        except TimeoutExpiredError:
            LOGGER.warning("Checking pod log")
            pod = get_model_catalog_pod()
            log = pod.log(container="catalog")
            if f"Listed {expected_num_models_from_hf_api} models from author" in log:
                LOGGER.warning("Found relevant log entry")
            else:
                LOGGER.warning(f"No relevant log entry: {log}")
                raise
