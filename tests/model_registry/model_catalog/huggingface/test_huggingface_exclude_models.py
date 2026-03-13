from typing import Self

import pytest
from ocp_resources.config_map import ConfigMap
from simple_logger.logger import get_logger

from tests.model_registry.model_catalog.utils import get_hf_catalog_str, get_models_from_catalog_api

LOGGER = get_logger(name=__name__)

pytestmark = [
    pytest.mark.skip_on_disconnected,
    pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace"),
]


@pytest.mark.parametrize(
    "updated_catalog_config_map, expected_models, excluded_models",
    [
        pytest.param(
            {
                "sources_yaml": get_hf_catalog_str(ids=["mixed"], excluded_models=["meta-llama/*", "ibm-granite/*"]),
            },
            ["microsoft/phi-2", "microsoft/Phi-4-mini-reasoning", "microsoft/Phi-3.5-mini-instruct"],
            ["meta-llama", "ibm-granite"],
            id="test_model_exclusion_wildcard_prefix",
            marks=pytest.mark.install,
        ),
        pytest.param(
            {
                "sources_yaml": get_hf_catalog_str(ids=["granite"], excluded_models=["ibm-granite/granite-4.0-h*"]),
            },
            ["ibm-granite/granite-4.0-micro", "ibm-granite/granite-4.0-micro-base"],
            ["ibm-granite/granite-4.0-h*"],
            id="test_model_exclusion_wildcard_suffix",
            marks=pytest.mark.install,
        ),
        pytest.param(
            {
                "sources_yaml": get_hf_catalog_str(
                    ids=["mixed"], excluded_models=["ibm-granite/granite-4.0-h-1b", "microsoft/phi-2"]
                ),
            },
            [
                "meta-llama/Llama-3.1-8B-Instruct",
                "microsoft/Phi-4-mini-reasoning",
                "microsoft/Phi-3.5-mini-instruct",
            ],
            ["ibm-granite/granite-4.0-h-1b", "microsoft/phi-2"],
            id="test_model_exclusion_specific_models",
            marks=(pytest.mark.install),
        ),
    ],
    indirect=["updated_catalog_config_map"],
)
@pytest.mark.usefixtures("updated_catalog_config_map")
class TestHuggingFaceModelExclusion:
    """Test HuggingFace model exclusion functionality"""

    def test_excluded_models_not_in_catalog(
        self: Self,
        updated_catalog_config_map: tuple[ConfigMap, str, str],
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        expected_models: list[str],
        excluded_models: list[str],
    ):
        """
        Test that excluded models do not appear in the catalog API response
        """
        LOGGER.info("Testing HuggingFace model exclusion functionality")

        # Get all models from the catalog API
        response = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
        )
        assert response["items"], "Expected models to be present in response"

        # Extract model names from API response
        catalog_model_names = [model.get("name", "") for model in response["items"]]
        assert set(expected_models).issubset(set(catalog_model_names)), (
            f"Expected {expected_models} models to be present in response. Found {catalog_model_names}"
        )
        LOGGER.info(f"With exclusion {excluded_models}, following models were found: {catalog_model_names}")
        violating_models = [
            model for model in catalog_model_names if any(model.startswith(prefix) for prefix in excluded_models)
        ]

        assert not violating_models, f"Found models with excluded prefixes: {violating_models}"
        LOGGER.info(f"Successfully verified {len(expected_models)} expected models are present")
