from collections.abc import Generator
from typing import Self

import pytest
from huggingface_hub import HfApi
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from simple_logger.logger import get_logger

from tests.model_registry.model_catalog.constants import HF_MODELS, HF_SOURCE_ID
from tests.model_registry.model_catalog.huggingface.utils import (
    assert_huggingface_values_matches_model_catalog_api_values,
    get_huggingface_model_from_api,
    wait_for_hugging_face_model_import,
    wait_for_huggingface_retrival_match,
    wait_for_last_sync_update,
)
from tests.model_registry.model_catalog.utils import (
    get_hf_catalog_str,
)

LOGGER = get_logger(name=__name__)

pytestmark = [pytest.mark.skip_on_disconnected]


class TestLastSyncedMetadataValidation:
    """Test HuggingFace model last synced timestamp validation"""

    @pytest.mark.parametrize(
        "updated_catalog_config_map_scope_function, initial_last_synced_values, model_name",
        [
            pytest.param(
                """
catalogs:
  - name: HuggingFace Hub
    id: hf_id
    type: hf
    enabled: true
    includedModels:
    - microsoft/phi-2
    properties:
      syncInterval: '2m'
""",
                "microsoft/phi-2",
                "microsoft/phi-2",
                id="test_hf_last_synced_custom",
            ),
        ],
        indirect=["updated_catalog_config_map_scope_function", "initial_last_synced_values"],
    )
    def test_huggingface_last_synced_custom(
        self: Self,
        updated_catalog_config_map_scope_function: Generator[ConfigMap],
        initial_last_synced_values: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        model_name: str,
    ):
        """
        Custom test for HuggingFace model last synced validation
        """
        # Get the model name from the parametrized test
        wait_for_last_sync_update(
            model_registry_rest_headers=model_registry_rest_headers,
            model_catalog_rest_url=model_catalog_rest_url,
            model_name=model_name,
            source_id="hf_id",
            initial_last_synced_values=float(initial_last_synced_values),
        )


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
@pytest.mark.usefixtures("epoch_time_before_config_map_update", "updated_catalog_config_map")
class TestHuggingFaceModelValidation:
    """Test HuggingFace model values by comparing values between HF API calls and Model Catalog api call"""

    def test_huggingface_model_metadata_last_synced(
        self: Self,
        epoch_time_before_config_map_update: float,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        expected_catalog_values: dict[str, str],
        huggingface_api: HfApi,
    ):
        """
        Validate HuggingFace model last synced timestamp is properly updated
        """
        LOGGER.info(
            f"Validating HuggingFace model last synced timestamps with {epoch_time_before_config_map_update} "
            "epoch (milliseconds)"
        )
        error = {}
        for model_name in expected_catalog_values:
            result = get_huggingface_model_from_api(
                model_catalog_rest_url=model_catalog_rest_url,
                model_registry_rest_headers=model_registry_rest_headers,
                model_name=model_name,
                source_id=HF_SOURCE_ID,
            )
            error_msg = ""
            if result["name"] != model_name:
                error_msg += f"Expected model name {model_name}, but got {result['name']}. "

            # Extract last_synced timestamp
            last_synced = result["customProperties"]["last_synced"]["string_value"]
            LOGGER.info(f"Model {model_name} last synced at: {last_synced}")

            # Validate that last_synced field exists and is not empty
            if not last_synced or last_synced == "":
                error_msg += f"last_synced field is not present for model {model_name}. "
            elif epoch_time_before_config_map_update > float(last_synced):
                error_msg += (
                    f"Model {model_name} last_synced ({last_synced}) should be after "
                    f"test start time ({epoch_time_before_config_map_update}). "
                )
            if error_msg:
                error[model_name] = error_msg
        if error:
            LOGGER.error(error)
            pytest.fail("Last synced validation failed")

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
        "updated_catalog_config_map_scope_function, num_models_from_hf_api_with_matching_criteria",
        [
            pytest.param(
                """
catalogs:
  - name: HuggingFace Hub
    id: hf_id
    type: hf
    enabled: true
    includedModels:
    - huggingface-course/*
""",
                {"org_name": "huggingface-course", "excluded_str": None},
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
      allowedOrganization: "huggingface-course"
    includedModels:
    - '*'
""",
                {"org_name": "huggingface-course", "excluded_str": None},
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
      allowedOrganization: "huggingface-course"
    includedModels:
    - '*'
    excludedModels:
    - '*-accelerate'
""",
                {"org_name": "huggingface-course", "excluded_str": "-accelerate"},
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
        admin_client: DynamicClient,
        model_registry_namespace: str,
        updated_catalog_config_map_scope_function: Generator[ConfigMap],
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        huggingface_api: bool,
        num_models_from_hf_api_with_matching_criteria: int,
    ):
        """
        Test that excluded models do not appear in the catalog API response
        """
        LOGGER.info("Testing HuggingFace model exclusion functionality")
        wait_for_hugging_face_model_import(
            admin_client=admin_client,
            model_registry_namespace=model_registry_namespace,
            hf_id="hf_id",
            expected_num_models_from_hf_api=num_models_from_hf_api_with_matching_criteria,
        )
        wait_for_huggingface_retrival_match(
            source_id="hf_id",
            model_registry_rest_headers=model_registry_rest_headers,
            model_catalog_rest_url=model_catalog_rest_url,
            expected_num_models_from_hf_api=num_models_from_hf_api_with_matching_criteria,
        )
