import pytest
import re
from simple_logger.logger import get_logger
from typing import Self

from ocp_resources.config_map import ConfigMap
from tests.model_registry.model_catalog.utils import (
    assert_source_error_state_message,
)
from tests.model_registry.model_catalog.huggingface.utils import assert_accessible_models_via_catalog_api
from tests.model_registry.utils import execute_get_command
from kubernetes.dynamic.exceptions import ResourceNotFoundError


LOGGER = get_logger(name=__name__)
INACCESSIBLE_MODELS: list[str] = [
    "jonburdo/private-test-model-1",
]
ACCESSIBLE_MODELS: list[str] = ["jonburdo/public-test-model-1", "jonburdo/test2", "jonburdo/gated-test-model-1"]
SOURCE_ID: str = "mixed_models_catalog"


@pytest.mark.parametrize(
    "updated_catalog_config_map",
    [
        pytest.param(
            {
                "sources_yaml": f"""
catalogs:
  - name: HuggingFace Mixed Models
    id: {SOURCE_ID}
    type: hf
    enabled: true
    includedModels:
      - "jonburdo/test2"
      - "jonburdo/public-test-model-1"
      - "jonburdo/private-test-model-1"
      - "jonburdo/gated-test-model-1"
"""
            },
            id="test_mixed_accessible_and_inaccessible_models",
        ),
    ],
    indirect=["updated_catalog_config_map"],
)
@pytest.mark.usefixtures(
    "updated_dsc_component_state_scope_session",
    "model_registry_namespace",
)
class TestHuggingFaceSourceErrorValidation:
    """Test cases for RHOAIENG-47934 - Partial model fetching errors should not affect other models."""

    def test_source_error_state_and_message(
        self: Self,
        updated_catalog_config_map: ConfigMap,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        RHOAIENG-47934: Verify source shows error state with correct error message.

        This test verifies that:
        1. The source is in error state due to private model fetch failure
        2. The error message contains the specific failed models
        """
        # Construct expected error message with failed models
        failed_models_str = ", ".join(INACCESSIBLE_MODELS)
        expected_error_message = (
            "Failed to fetch some models, ensure models exist and are accessible with "
            f"given credentials. Failed models: [{failed_models_str}]"
        )

        LOGGER.info(f"Testing source error state for failed models: {INACCESSIBLE_MODELS}")

        assert_source_error_state_message(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            expected_error_message=expected_error_message,
            source_id=SOURCE_ID,
        )

    @pytest.mark.xfail(
        reason="RHOAIENG-49162: API call using source_label does not find models, when source is in error state"
    )
    def test_accessible_models_catalog_api_source_id(
        self: Self,
        updated_catalog_config_map: ConfigMap,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        RHOAIENG-49162: Check that accessible models are visible through catalog API using source label.

        This test verifies that accessible models are still returned by the catalog API
        even when the source is in error state.
        """
        assert_accessible_models_via_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            expected_accessible_models=ACCESSIBLE_MODELS,
            source_label=SOURCE_ID,  # Filters by specific source
        )

    def test_accessible_models_catalog_api_no_source_id(
        self: Self,
        updated_catalog_config_map: ConfigMap,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        RHOAIENG-47934: Check that accessible models are visible through catalog API.

        This test verifies that accessible models are still returned by the catalog API
        even when the source is in error state.
        """
        assert_accessible_models_via_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            expected_accessible_models=ACCESSIBLE_MODELS,
            source_label=None,  # Searches all sources
        )

    def test_inaccessible_models_not_found_via_api_calls(
        self: Self,
        updated_catalog_config_map: ConfigMap,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        RHOAIENG-47934: Ensure inaccessible models are not found via API calls.

        This test verifies that inaccessible models (private/gated) correctly return
        "Not Found" errors when accessed via individual model API endpoints.
        """
        error_pattern = r"No model found '([^']+)' in source '([^']+)'"
        LOGGER.info(f"Testing that inaccessible models return 'Not Found': {INACCESSIBLE_MODELS}")

        for model_name in INACCESSIBLE_MODELS:
            with pytest.raises(ResourceNotFoundError) as exc_info:
                execute_get_command(
                    url=f"{model_catalog_rest_url[0]}sources/{SOURCE_ID}/models/{model_name}",
                    headers=model_registry_rest_headers,
                )
            match = re.search(error_pattern, str(exc_info.value))
            assert match.group(1) == model_name, f"Expected model '{model_name}' in error, got '{match.group(1)}'"
            assert match.group(2) == SOURCE_ID, f"Expected source '{SOURCE_ID}' in error, got '{match.group(2)}'"
