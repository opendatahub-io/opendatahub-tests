from typing import Self

import pytest
import requests
import structlog

from tests.ai_hub.utils import execute_authenticated_post

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.tier3
@pytest.mark.usefixtures("model_registry_namespace")
class TestMCPPreviewErrorHandling:
    """
    Test class for validating the MCP server preview API error handling.
    """

    @pytest.mark.parametrize(
        "config_content, expected_error_message",
        [
            pytest.param(
                "assetType: mcp_servers\n",
                "missing required field: type",
                id="test_missing_type",
            ),
            pytest.param(
                "assetType: mcp_servers\ntype: unsupported-type\nproperties:\n  somePath: /some/path\n",
                "unsupported source type for MCP preview: unsupported-type",
                id="test_unsupported_type",
            ),
            pytest.param(
                "assetType: mcp_servers\n"
                "type: yaml\n"
                "properties:\n"
                "  yamlCatalogPath: /nonexistent/path.yaml\n"
                "includedServers:\n"
                '  - "*"\n',
                "/nonexistent/path.yaml: no such file or directory",
                id="test_nonexistent_path",
            ),
            pytest.param(
                "assetType: mcp_servers\ninvalid-yaml-syntax:\n  - this: is: broken::\n",
                "failed to parse config:",
                id="test_invalid_yaml",
            ),
        ],
    )
    def test_mcp_preview_invalid_config(
        self: Self,
        model_catalog_preview_url: str,
        preview_user_token: str,
        config_content: str,
        expected_error_message: str,
    ):
        """
        Test that the MCP preview API returns 422 with appropriate error messages for invalid configurations.

        Given an invalid MCP preview config,
        When posting to the preview endpoint,
        Then the API should return 422 Unprocessable Entity with a descriptive error message.
        """
        url = f"{model_catalog_preview_url}sources/preview?pageSize=100"

        files = {"config": ("config.yaml", config_content, "application/x-yaml")}

        with pytest.raises(requests.exceptions.HTTPError) as exc_info:
            execute_authenticated_post(url=url, token=preview_user_token, files=files)

        assert exc_info.value.response.status_code == 422, f"Expected 422, got {exc_info.value.response.status_code}"

        error_message = exc_info.value.response.json().get("message", "")
        assert expected_error_message in error_message, (
            f"Expected error to contain '{expected_error_message}', got: {error_message}"
        )

        LOGGER.info(f"Correctly received 422 with message: {error_message}")
