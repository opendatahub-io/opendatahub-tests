from tests.model_registry.model_catalog.constants import (
    EXPECTED_CUSTOM_CATALOG_VALUES,
    CUSTOM_CATALOG_ID1,
    SAMPLE_MODEL_NAME1,
    CUSTOM_CATALOG_ID2,
    SAMPLE_MODEL_NAME2,
    MULTIPLE_CUSTOM_CATALOG_VALUES,
    SAMPLE_MODEL_NAME3,
)
from ocp_resources.config_map import ConfigMap
import pytest
from simple_logger.logger import get_logger
from typing import Self

from tests.model_registry.model_catalog.utils import (
    execute_get_command,
    get_catalog_str,
    get_sample_yaml_str,
    ResourceNotFoundError,
)

LOGGER = get_logger(name=__name__)


@pytest.mark.parametrize(
    "updated_catalog_config_map, expected_catalog_values",
    [
        pytest.param(
            {
                "sources_yaml": get_catalog_str(ids=[CUSTOM_CATALOG_ID1]),
                "sample_yaml": {"sample-custom-catalog1.yaml": get_sample_yaml_str(models=[SAMPLE_MODEL_NAME1])},
            },
            EXPECTED_CUSTOM_CATALOG_VALUES,
            id="test_file_test_catalog",
        ),
        pytest.param(
            {
                "sources_yaml": get_catalog_str(ids=[CUSTOM_CATALOG_ID1, CUSTOM_CATALOG_ID2]),
                "sample_yaml": {
                    "sample-custom-catalog1.yaml": get_sample_yaml_str(models=[SAMPLE_MODEL_NAME1]),
                    "sample-custom-catalog2.yaml": get_sample_yaml_str(models=[SAMPLE_MODEL_NAME2]),
                },
            },
            MULTIPLE_CUSTOM_CATALOG_VALUES,
            id="test_file_test_catalog_multiple_sources",
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures(
    "model_registry_namespace",
    "updated_catalog_config_map",
)
class TestModelCatalogCustom:
    def test_model_custom_catalog_list_sources(
        self: Self,
        updated_catalog_config_map: tuple[ConfigMap, str, str],
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        expected_catalog_values: dict[str, str],
    ):
        """
        Validate sources api for model catalog
        """
        url = f"{model_catalog_rest_url[0]}sources"
        results = execute_get_command(
            url=url,
            headers=model_registry_rest_headers,
        )["items"]
        ids_from_query = [result_entry["id"] for result_entry in results]
        ids_expected = [expected_entry["id"] for expected_entry in expected_catalog_values]
        assert set(ids_expected).issubset(set(ids_from_query)), (
            f"Expected model catalogs: {expected_catalog_values}. Actual model catalogs: {results}"
        )

    def test_model_custom_catalog_get_models_by_source(
        self: Self,
        updated_catalog_config_map: tuple[ConfigMap, str, str],
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        expected_catalog_values: dict[str, str],
    ):
        """
        Validate models api for model catalog associated with a specific source
        """
        for expected_entry in expected_catalog_values:
            url = f"{model_catalog_rest_url[0]}models?source={expected_entry['id']}"
            result = execute_get_command(
                url=url,
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
        for expected_entry in expected_catalog_values:
            model_name = expected_entry["model_name"]
            url = f"{model_catalog_rest_url[0]}sources/{expected_entry['id']}/models/{model_name}"
            result = execute_get_command(
                url=url,
                headers=model_registry_rest_headers,
            )
            assert result["name"] == model_name

    def test_model_custom_catalog_get_model_artifact(
        self: Self,
        updated_catalog_config_map: tuple[ConfigMap, str, str],
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        expected_catalog_values: dict[str, str],
    ):
        """
        Get Model artifacts for model associated with specific source
        """
        for expected_entry in expected_catalog_values:
            model_name = expected_entry["model_name"]
            url = f"{model_catalog_rest_url[0]}sources/{expected_entry['id']}/models/{model_name}/artifacts"

            artifacts = execute_get_command(
                url=url,
                headers=model_registry_rest_headers,
            )["items"]

            assert artifacts, f"No artifacts found for {model_name}"
            assert artifacts[0]["uri"]

    @pytest.mark.dependency(name="test_model_custom_catalog_add_model")
    def test_model_custom_catalog_add_model(
        self: Self,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        expected_catalog_values: dict[str, str],
        update_configmap_data_add_model: dict[str, str],
    ):
        """
        Add a model to a source and ensure it is added to the catalog
        """
        url = f"{model_catalog_rest_url[0]}sources/{CUSTOM_CATALOG_ID1}/models/{SAMPLE_MODEL_NAME3}"
        result = execute_get_command(
            url=url,
            headers=model_registry_rest_headers,
        )
        assert result["name"] == SAMPLE_MODEL_NAME3

    @pytest.mark.dependency(depends=["test_model_custom_catalog_add_model"])
    def test_model_custom_catalog_remove_model(
        self: Self,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        expected_catalog_values: dict[str, str],
    ):
        """
        Ensure models are removed from the catalog
        """
        url = f"{model_catalog_rest_url[0]}sources/{CUSTOM_CATALOG_ID1}/models/{SAMPLE_MODEL_NAME3}"
        with pytest.raises(ResourceNotFoundError):
            result = execute_get_command(
                url=url,
                headers=model_registry_rest_headers,
            )
            LOGGER.info(f"URL: {url} Result: {result}")
