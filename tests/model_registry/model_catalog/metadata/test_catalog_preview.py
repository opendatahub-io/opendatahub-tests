from typing import Self

import pytest
import requests
import yaml
from simple_logger.logger import get_logger

from tests.model_registry.model_catalog.constants import VALIDATED_CATALOG_FILE, VALIDATED_CATALOG_ID
from tests.model_registry.model_catalog.metadata.utils import (
    build_catalog_preview_config,
    execute_model_catalog_post_command,
    validate_catalog_preview_counts,
    validate_catalog_preview_items,
)

LOGGER = get_logger(name=__name__)


class TestCatalogPreviewExistingSource:
    """
    Test class for validating the catalog preview API for an existing source
    RHOAIENG-41826
    """

    @pytest.mark.parametrize("default_model_catalog_yaml_content", [VALIDATED_CATALOG_ID], indirect=True)
    @pytest.mark.usefixtures(
        "model_registry_namespace",
    )
    def test_catalog_preview_included_and_excluded_models_filters(
        self: Self,
        model_catalog_rest_url: list[str],
        user_token_for_api_calls: str,
        default_model_catalog_yaml_content: dict,
    ):
        """
        Test the catalog preview API for an existing source with includedModels and excludedModels filters.

        This test validates that the preview endpoint correctly:
        - Includes models matching the includedModels pattern (RedHatAI/*)
        - Excludes models matching the excludedModels pattern (*-quantized*)
        - Returns accurate summary counts matching the actual YAML content on the pod
        """
        # Define filter patterns (used both in config and for validation)
        included_patterns = ["RedHatAI/*"]
        excluded_patterns = ["*-quantized*"]

        # Create config.yaml content with the filter patterns
        config_content = build_catalog_preview_config(
            yaml_catalog_path=VALIDATED_CATALOG_FILE,
            included_patterns=included_patterns,
            excluded_patterns=excluded_patterns,
        )

        url = f"{model_catalog_rest_url[0]}sources/preview?pageSize=100"
        LOGGER.info(f"Testing preview endpoint: {url}")
        LOGGER.info(f"Config content:\n{config_content}")

        # Execute preview command with multipart/form-data
        files = {"config": ("config.yaml", config_content, "application/x-yaml")}
        result = execute_model_catalog_post_command(
            url=url,
            token=user_token_for_api_calls,
            files=files,
        )

        # Validate API counts against YAML content
        summary = result.get("summary")
        assert summary is not None, f"Missing 'summary' field in API response: {result}"
        yaml_models = default_model_catalog_yaml_content.get("models", [])
        validate_catalog_preview_counts(
            api_counts=summary,
            yaml_models=yaml_models,
            included_patterns=included_patterns,
            excluded_patterns=excluded_patterns,
        )

        # Validate that each item has correct 'included' property
        validate_catalog_preview_items(
            result=result,
            included_patterns=included_patterns,
            excluded_patterns=excluded_patterns,
        )

    @pytest.mark.parametrize("default_model_catalog_yaml_content", [VALIDATED_CATALOG_ID], indirect=True)
    @pytest.mark.usefixtures(
        "model_registry_namespace",
    )
    def test_catalog_preview_no_filters(
        self: Self,
        model_catalog_rest_url: list[str],
        user_token_for_api_calls: str,
        default_model_catalog_yaml_content: dict,
    ):
        """
        Test the catalog preview API for an existing source with no filters.

        This test validates that the preview endpoint correctly returns all models when no filters are provided.
        """
        config_content = build_catalog_preview_config(yaml_catalog_path=VALIDATED_CATALOG_FILE)

        url = f"{model_catalog_rest_url[0]}sources/preview?pageSize=100"
        LOGGER.info(f"Testing preview endpoint: {url}")
        LOGGER.info(f"Config content:\n{config_content}")

        # Execute preview command with multipart/form-data
        files = {"config": ("config.yaml", config_content, "application/x-yaml")}
        result = execute_model_catalog_post_command(
            url=url,
            token=user_token_for_api_calls,
            files=files,
        )

        # Validate API counts against YAML content (no filters = all included)
        summary = result.get("summary")
        assert summary is not None, f"Missing 'summary' field in API response: {result}"
        yaml_models = default_model_catalog_yaml_content.get("models", [])
        validate_catalog_preview_counts(
            api_counts=summary,
            yaml_models=yaml_models,
            included_patterns=None,
            excluded_patterns=None,
        )

        # Validate that each item has correct 'included' property (all should be included)
        validate_catalog_preview_items(
            result=result,
            included_patterns=None,
            excluded_patterns=None,
        )

    @pytest.mark.parametrize("default_model_catalog_yaml_content", [VALIDATED_CATALOG_ID], indirect=True)
    @pytest.mark.usefixtures(
        "model_registry_namespace",
    )
    @pytest.mark.parametrize(
        "filter_status",
        [
            pytest.param("all", id="filter_all"),
            pytest.param("included", id="filter_included"),
            pytest.param("excluded", id="filter_excluded"),
        ],
    )
    def test_catalog_preview_filter_status(
        self: Self,
        model_catalog_rest_url: list[str],
        user_token_for_api_calls: str,
        default_model_catalog_yaml_content: dict,
        filter_status: str,
    ):
        """
        Test the catalog preview API with filterStatus parameter.

        This test validates that the filterStatus parameter correctly filters items:
        - filterStatus=all returns all models (both included and excluded)
        - filterStatus=included returns only items with included: true
        - filterStatus=excluded returns only items with included: false
        """
        # Define filter patterns to ensure we have both included and excluded models
        included_patterns = ["RedHatAI/*"]
        excluded_patterns = ["*-quantized*"]

        # Create config.yaml content with the filter patterns
        config_content = build_catalog_preview_config(
            yaml_catalog_path=VALIDATED_CATALOG_FILE,
            included_patterns=included_patterns,
            excluded_patterns=excluded_patterns,
        )

        url = f"{model_catalog_rest_url[0]}sources/preview?pageSize=100&filterStatus={filter_status}"
        LOGGER.info(f"Testing preview endpoint with filterStatus={filter_status}: {url}")

        # Execute preview command with multipart/form-data
        files = {"config": ("config.yaml", config_content, "application/x-yaml")}
        result = execute_model_catalog_post_command(
            url=url,
            token=user_token_for_api_calls,
            files=files,
        )

        # Validate items match the filter status
        validate_catalog_preview_items(
            result=result, included_patterns=included_patterns, excluded_patterns=excluded_patterns
        )

        # Validate counts based on filter_status
        summary = result.get("summary")
        assert summary is not None, f"Missing 'summary' field in API response: {result}"
        items = result.get("items", [])

        if filter_status == "all":
            assert len(items) == summary["totalModels"], (
                f"Total items ({len(items)}) doesn't match totalModels count ({summary['totalModels']})"
            )
            LOGGER.info(f"filterStatus=all validation passed: {len(items)} total items")
        else:
            expected_count = summary["includedModels"] if filter_status == "included" else summary["excludedModels"]
            assert len(items) == expected_count, (
                f"Number of items ({len(items)}) doesn't match {filter_status}Models count ({expected_count})"
            )
            LOGGER.info(f"filterStatus={filter_status} validation passed: {len(items)} items")


class TestCatalogPreviewErrorHandling:
    """
    Test class for validating the catalog preview API error handling
    RHOAIENG-41826
    """

    @pytest.mark.parametrize(
        "config_content, expected_status_code, expected_error_message",
        [
            pytest.param(
                """type: yaml
properties:
  yamlCatalogPath: /nonexistent/path/catalog.yaml
includedModels:
  - "*"
""",
                422,
                "/nonexistent/path/catalog.yaml: no such file or directory",
                id="nonexistent_yaml_path",
            ),
            pytest.param(
                """invalid-yaml-syntax:
  - this: is: broken::
""",
                422,
                "failed to parse config:",
                id="invalid_yaml_syntax",
            ),
            pytest.param(
                """type: unsupported-type
properties:
  somePath: /some/path
""",
                422,
                "unsupported source type: unsupported-type",
                id="unsupported_catalog_type",
            ),
        ],
    )
    @pytest.mark.usefixtures(
        "model_registry_namespace",
    )
    def test_catalog_preview_invalid_config(
        self: Self,
        model_catalog_rest_url: list[str],
        user_token_for_api_calls: str,
        config_content: str,
        expected_status_code: int,
        expected_error_message: str,
    ):
        """
        Test that the catalog preview API returns appropriate error codes and messages for invalid configurations.

        This test validates that the preview endpoint correctly handles:
        - Nonexistent YAML file paths (returns 422 with file path error)
        - Invalid YAML syntax (returns 422 with parse error)
        - Unsupported catalog types (returns 422 with type error)
        """
        url = f"{model_catalog_rest_url[0]}sources/preview?pageSize=100"
        LOGGER.info(f"Testing preview endpoint with invalid config: {url}")

        files = {"config": ("config.yaml", config_content, "application/x-yaml")}

        with pytest.raises(requests.exceptions.HTTPError) as exc_info:
            execute_model_catalog_post_command(
                url=url,
                token=user_token_for_api_calls,
                files=files,
            )

        assert exc_info.value.response.status_code == expected_status_code, (
            f"Expected status code {expected_status_code}, got {exc_info.value.response.status_code}"
        )

        error_response = exc_info.value.response.json()
        error_message = error_response.get("message", "")

        assert expected_error_message in error_message, (
            f"Expected error message to contain '{expected_error_message}', got: {error_message}"
        )

        LOGGER.info(
            f"Correctly received error status {exc_info.value.response.status_code} with message: {error_message}"
        )


class TestCatalogPreviewUserProvidedData:
    """
    Test class for validating the catalog preview API with user-provided catalog data
    RHOAIENG-41826
    """

    @pytest.mark.usefixtures(
        "model_registry_namespace",
    )
    def test_catalog_preview_with_custom_catalog_data(
        self: Self,
        model_catalog_rest_url: list[str],
        user_token_for_api_calls: str,
    ):
        """
        Test the catalog preview API with user-provided catalog data.

        This test validates that the preview endpoint accepts and processes custom catalog data
        provided via the catalogData parameter, applying filters correctly.
        """
        # Define filter patterns
        included_patterns = ["ibm-granite/*"]
        excluded_patterns = ["*-experimental"]

        # Create config using user-provided data
        config_content = build_catalog_preview_config(
            included_patterns=included_patterns,
            excluded_patterns=excluded_patterns,
        )

        # Create user-provided catalog data with known models
        catalog_data = """models:
  - name: ibm-granite/granite-3.0-8b-instruct
    description: Granite 8B Instruct model
  - name: ibm-granite/granite-3.0-2b-draft
    description: Draft version
  - name: meta-llama/llama-3-8b
    description: Llama 3 model
  - name: mistral/mistral-7b
    description: Mistral model
  - name: meta-llama/llama-2-experimental
    description: Experimental version
"""

        url = f"{model_catalog_rest_url[0]}sources/preview?pageSize=100"
        LOGGER.info(f"Testing preview endpoint with user-provided catalog data: {url}")

        # Execute preview command with both config and catalogData
        files = {
            "config": ("config.yaml", config_content, "application/x-yaml"),
            "catalogData": ("catalog-data.yaml", catalog_data, "application/x-yaml"),
        }
        result = execute_model_catalog_post_command(
            url=url,
            token=user_token_for_api_calls,
            files=files,
        )

        # Create expected model data for validation
        yaml_models = yaml.safe_load(catalog_data).get("models", [])

        # Validate API counts against provided YAML data
        summary = result.get("summary")
        assert summary is not None, f"Missing 'summary' field in API response: {result}"
        validate_catalog_preview_counts(
            api_counts=summary,
            yaml_models=yaml_models,
            included_patterns=included_patterns,
            excluded_patterns=excluded_patterns,
        )

        # Validate that each item has correct 'included' property
        validate_catalog_preview_items(
            result=result,
            included_patterns=included_patterns,
            excluded_patterns=excluded_patterns,
        )

        LOGGER.info("User-provided catalog data validation passed")
