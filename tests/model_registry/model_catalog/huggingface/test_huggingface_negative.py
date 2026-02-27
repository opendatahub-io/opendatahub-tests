from typing import Self

import pytest
from ocp_resources.config_map import ConfigMap
from simple_logger.logger import get_logger

from tests.model_registry.model_catalog.utils import assert_source_error_state_message

LOGGER = get_logger(name=__name__)

pytestmark = [
    pytest.mark.skip_on_disconnected,
    pytest.mark.usefixtures(
        "updated_dsc_component_state_scope_session",
        "model_registry_namespace",
    ),
]


@pytest.mark.tier3
class TestHuggingFaceNegative:
    @pytest.mark.parametrize(
        "updated_catalog_config_map_scope_function, expected_error_message",
        [
            pytest.param(
                """
catalogs:
  - name: HuggingFace Hub
    id: error_catalog
    type: hf
    enabled: true
""",
                "includedModels cannot be empty for Hugging Face catalog",
                id="test_hf_source_no_include_model",
            ),
            pytest.param(
                """
catalogs:
  - name: HuggingFace Hub
    id: error_catalog
    type: hf
    enabled: true
    includedModels:
    - abc-random*
""",
                'failed to expand model patterns: wildcard pattern "abc-random*" is not supported - '
                "Hugging Face requires a specific organization",
                id="test_hf_source_invalid_organization",
            ),
            pytest.param(
                """
catalogs:
  - name: HuggingFace Hub
    id: error_catalog
    type: hf
    enabled: true
    includedModels:
    - '*'
""",
                'failed to expand model patterns: wildcard pattern "*" is not supported - '
                "Hugging Face requires a specific organization",
                id="test_hf_source_global_wildcard",
            ),
            pytest.param(
                """
catalogs:
  - name: HuggingFace Hub
    id: error_catalog
    type: hf
    enabled: true
    includedModels:
    - '*/*'
""",
                'failed to expand model patterns: wildcard pattern "*/*" is not supported - '
                "Hugging Face requires a specific organization",
                id="test_hf_source_global_org_wildcard",
            ),
            pytest.param(
                """
catalogs:
  - name: HuggingFace Hub
    id: error_catalog
    type: hf
    enabled: true
    includedModels:
    - 'RedHatAI/'
""",
                'failed to expand model patterns: wildcard pattern "RedHatAI/" is not supported - '
                "Hugging Face requires a specific organization",
                id="test_hf_source_empty_model_name",
            ),
            pytest.param(
                """
catalogs:
  - name: HuggingFace Hub
    id: error_catalog
    type: hf
    enabled: true
    includedModels:
    - '*RedHatAI*'
""",
                'failed to expand model patterns: wildcard pattern "*RedHatAI*" is not supported - '
                "Hugging Face requires a specific organization",
                id="test_hf_source_multiple_wildcards",
            ),
            pytest.param(
                """
catalogs:
  - name: HuggingFace Hub
    id: error_catalog
    type: hf
    enabled: true
    properties:
      allowedOrganization: "abc-random"
    includedModels:
    - '*'
""",
                "failed to expand model patterns: no models found",
                id="test_hf_source_non_existent_allowed_organization",
            ),
            pytest.param(
                """
catalogs:
    - name: HuggingFace Hub
      id: error_catalog
      type: hf
      enabled: true
      includedModels:
      - 'microsoft/phi-3-abc-random'
""",
                "Failed to fetch some models, ensure models exist and are accessible with given credentials. "
                "Failed models: [microsoft/phi-3-abc-random]",
                id="test_hf_bad_model_name",
            ),
        ],
        indirect=["updated_catalog_config_map_scope_function"],
    )
    def test_huggingface_source_error_state(
        self: Self,
        updated_catalog_config_map_scope_function: ConfigMap,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        expected_error_message: str,
    ):
        assert_source_error_state_message(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            expected_error_message=expected_error_message,
            source_id="error_catalog",
        )
