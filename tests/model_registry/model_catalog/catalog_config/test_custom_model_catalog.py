from typing import Self

import pytest
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.config_map import ConfigMap
from simple_logger.logger import get_logger

from tests.model_registry.constants import CUSTOM_CATALOG_ID1, SAMPLE_MODEL_NAME1
from tests.model_registry.model_catalog.constants import (
    CUSTOM_CATALOG_ID2,
    EXPECTED_CUSTOM_CATALOG_VALUES,
    EXPECTED_HF_CATALOG_VALUES,
    EXPECTED_MULTIPLE_HF_CATALOG_VALUES,
    MULTIPLE_CUSTOM_CATALOG_VALUES,
    SAMPLE_MODEL_NAME2,
    SAMPLE_MODEL_NAME3,
)
from tests.model_registry.model_catalog.utils import get_hf_catalog_str
from tests.model_registry.utils import (
    execute_get_command,
    get_catalog_str,
    get_sample_yaml_str,
    validate_model_catalog_sources,
)

LOGGER = get_logger(name=__name__)


@pytest.mark.parametrize(
    "updated_catalog_config_map, expected_catalog_values, is_huggingface",
    [
        pytest.param(
            {
                "sources_yaml": get_hf_catalog_str(ids=["mixed"]),
            },
            EXPECTED_HF_CATALOG_VALUES,
            True,
            id="test_HF_catalog_single_source",
            marks=(pytest.mark.install),
        ),
        pytest.param(
            {
                "sources_yaml": get_hf_catalog_str(ids=["mixed", "granite"]),
            },
            EXPECTED_MULTIPLE_HF_CATALOG_VALUES,
            True,
            id="test_HF_catalog_multiple_sources",
            marks=(pytest.mark.install),
        ),
        pytest.param(
            {
                "sources_yaml": get_catalog_str(ids=[CUSTOM_CATALOG_ID1]),
                "sample_yaml": {"sample-custom-catalog1.yaml": get_sample_yaml_str(models=[SAMPLE_MODEL_NAME1])},
            },
            EXPECTED_CUSTOM_CATALOG_VALUES,
            False,
            id="test_file_test_catalog",
            marks=(pytest.mark.pre_upgrade, pytest.mark.post_upgrade, pytest.mark.install),
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
            False,
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
        is_huggingface: bool,
    ):
        """
        Validate sources api for model catalog
        """
        validate_model_catalog_sources(
            model_catalog_sources_url=f"{model_catalog_rest_url[0]}sources",
            rest_headers=model_registry_rest_headers,
            expected_catalog_values=expected_catalog_values,
        )

    def test_model_custom_catalog_get_models_by_source(
        self: Self,
        updated_catalog_config_map: tuple[ConfigMap, str, str],
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        expected_catalog_values: dict[str, str],
        is_huggingface: bool,
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
        is_huggingface: bool,
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
        skip_on_huggingface_source: None,
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
        is_huggingface: bool,
        skip_on_huggingface_source: None,
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
        is_huggingface: bool,
    ):
        """
        Ensure models are removed from the catalog
        """
        url = f"{model_catalog_rest_url[0]}sources/{CUSTOM_CATALOG_ID1}/models/{SAMPLE_MODEL_NAME3}"
        with pytest.raises(ResourceNotFoundError):
            execute_get_command(
                url=url,
                headers=model_registry_rest_headers,
            )
