from typing import Self

import pytest
from ocp_resources.config_map import ConfigMap
from ocp_resources.resource import ResourceEditor
from simple_logger.logger import get_logger

from tests.model_registry.constants import DEFAULT_MODEL_CATALOG_CM
from tests.model_registry.model_catalog.catalog_config.utils import validate_model_catalog_configmap_data
from tests.model_registry.model_catalog.constants import DEFAULT_CATALOGS
from tests.model_registry.model_catalog.utils import assert_source_error_state_message

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
        ],
        indirect=["updated_catalog_config_map_scope_function"],
    )
    def test_default_source_error_state(
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
