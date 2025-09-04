from tests.model_registry.model_catalog.constants import (
    CUSTOM_ECOSYSTEM_CATALOG,
    CUSTOM_CATALOG_WITH_FILE,
    SAMPLE_CATALOG_YAML,
    EXPECTED_CUSTOM_CATALOG_VALUES,
    EXPECTED_ECHO_CATALOG_VALUES,
)
from ocp_resources.config_map import ConfigMap
import pytest
from simple_logger.logger import get_logger
from typing import Self

from tests.model_registry.model_catalog.utils import execute_get_command

LOGGER = get_logger(name=__name__)


@pytest.mark.parametrize(
    "updated_catalog_config_map, expected_catalog_values",
    [
        pytest.param({"sources_yaml": CUSTOM_ECOSYSTEM_CATALOG}, EXPECTED_ECHO_CATALOG_VALUES, id="rhec_test_catalog"),
        pytest.param(
            {"sources_yaml": CUSTOM_CATALOG_WITH_FILE, "sample_yaml": SAMPLE_CATALOG_YAML},
            EXPECTED_CUSTOM_CATALOG_VALUES,
            id="file_test_catalog",
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures(
    "model_registry_namespace",
    "updated_catalog_config_map",
)
class TestModelCatalogCustom:
    def test_model_custom_catalog_sources(
        self: Self,
        updated_catalog_config_map: tuple[ConfigMap, str, str],
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        expected_catalog_values: dict[str, str],
    ):
        """
        Validate sources api for model catalog
        """
        result = execute_get_command(
            url=f"{model_catalog_rest_url[0]}sources",
            headers=model_registry_rest_headers,
        )["items"]
        assert len(result) == 1
        assert result[0]["id"] == expected_catalog_values["id"]

    def test_model_custom_catalog_models(
        self: Self,
        updated_catalog_config_map: tuple[ConfigMap, str, str],
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        expected_catalog_values: dict[str, str],
    ):
        """
        Validate models api for model catalog associated with a specific source
        """
        result = execute_get_command(
            url=f"{model_catalog_rest_url[0]}models?source={expected_catalog_values['id']}",
            headers=model_registry_rest_headers,
        )["items"]
        assert result, f"Expected custom models to be present. Actual: {result}"

    def test_model_custom_catalog_get_model_by_name(
        self: Self,
        updated_catalog_config_map: tuple[ConfigMap, str, str],
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        expected_catalog_values: dict[str, str],
    ):
        """
        Get Model by name associated with a specific source
        """
        model_name = expected_catalog_values["model_name"]
        result = execute_get_command(
            url=f"{model_catalog_rest_url[0]}sources/{expected_catalog_values['id']}/models/{model_name}",
            headers=model_registry_rest_headers,
        )
        assert result["name"] == model_name

    def test_model_custom_catalog_get_artifact(
        self: Self,
        updated_catalog_config_map: tuple[ConfigMap, str, str],
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        expected_catalog_values: dict[str, str],
    ):
        """
        Get Model artifacts for model associated with specific source
        """
        model_name = expected_catalog_values["model_name"]
        artifacts = execute_get_command(
            url=f"{model_catalog_rest_url[0]}sources/{expected_catalog_values['id']}/models/{model_name}/artifacts",
            headers=model_registry_rest_headers,
        )["items"]

        assert artifacts, f"No artifacts found for {model_name}"
        assert artifacts[0]["uri"]
