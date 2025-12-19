import pytest
from simple_logger.logger import get_logger
from typing import Self

from ocp_resources.config_map import ConfigMap
from ocp_resources.resource import ResourceEditor
from tests.model_registry.constants import DEFAULT_MODEL_CATALOG_CM
from tests.model_registry.model_catalog.constants import DEFAULT_CATALOGS
from tests.model_registry.model_catalog.utils import validate_model_catalog_configmap_data
from tests.model_registry.utils import execute_get_command

LOGGER = get_logger(name=__name__)

pytestmark = [
    pytest.mark.usefixtures(
        "updated_dsc_component_state_scope_session",
        "model_registry_namespace",
    )
]


@pytest.mark.skip_must_gather
class TestDefaultCatalogNegative:
    """Negative tests for default catalog configuration"""

    @pytest.mark.parametrize(
        "model_catalog_config_map, modified_sources_yaml",
        [
            pytest.param(
                {"configmap_name": DEFAULT_MODEL_CATALOG_CM},
                """
catalogs:
  - name: Modified Catalog
    id: modified_catalog
    type: yaml
    properties:
      yamlCatalogPath: /shared-data/modified-catalog.yaml
""",
                id="test_modify_catalog_structure",
            ),
        ],
        indirect=["model_catalog_config_map"],
    )
    def test_modify_default_catalog_configmap_reconciles(
        self: Self, model_catalog_config_map: ConfigMap, modified_sources_yaml: str
    ):
        """
        Test that attempting to modify the default catalog configmap raises an exception.
        This validates that the default catalog configmap is protected from modifications.
        """
        # Attempt to modify the configmap - this should raise an exception
        patches = {"data": {"sources.yaml": modified_sources_yaml}}

        with ResourceEditor(patches={model_catalog_config_map: patches}):
            # This block should raise an exception due to configmap protection
            LOGGER.info("Attempting to modify protected configmap")

        # Verify the configmap was not actually modified
        validate_model_catalog_configmap_data(
            configmap=model_catalog_config_map, num_catalogs=len(DEFAULT_CATALOGS.keys())
        )

    @pytest.mark.parametrize(
        "updated_catalog_config_map_scope_function, expected_error_message",
        [
            pytest.param(
                """
catalogs:
  - name: Modified Catalog
    id: error_catalog
    type: yaml
    properties:
      yamlCatalogPath: non-existent-catalog.yaml
""",
                "non-existent-catalog.yaml: no such file or directory",
                id="test_source_error_invalid_path",
            ),
            pytest.param(
                """
catalogs:
  - name: HuggingFace Hub
    id: error_catalog
    type: hf
    enabled: true
""",
                "includedModels cannot be empty for HuggingFace catalog",
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
    - ibm-granite/*
""",
                "HuggingFace API returned status 401",
                id="test_hf_source_unauthorized",
                marks=pytest.mark.xfail(reason="RHOAIENG-42213 is causing this failure"),
            ),
        ],
        indirect=["updated_catalog_config_map_scope_function"],
    )
    def test_source_error_state(
        self: Self,
        updated_catalog_config_map_scope_function: ConfigMap,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        expected_error_message,
    ):
        results = execute_get_command(
            url=f"{model_catalog_rest_url[0]}sources",
            headers=model_registry_rest_headers,
        )["items"]
        # pick the relevant source first by id:
        matched_source = [result for result in results if result["id"] == "error_catalog"]
        assert matched_source, f"Matched expected source not found: {results}"
        assert matched_source[0]["status"] == "error"
        assert expected_error_message in matched_source[0]["error"], (
            f"Expected error: {expected_error_message} not found in {matched_source[0]['error']}"
        )
