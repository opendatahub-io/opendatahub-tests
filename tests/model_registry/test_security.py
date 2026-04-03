from typing import Self

import pytest
import requests
import structlog
from ocp_resources.route import Route

from tests.model_registry.constants import MCP_CATALOG_API_PATH

LOGGER = structlog.get_logger(name=__name__)
MODEL_CATALOG_API_PATH = "/api/model_catalog/v1alpha1/"


@pytest.mark.tier1
class TestCatalogSecurity:
    """Tests for catalog endpoint security (TC-SEC-001)."""

    @pytest.mark.parametrize(
        "api_path",
        [
            pytest.param(f"{MCP_CATALOG_API_PATH}mcp_servers", id="mcp_catalog_mcp_servers"),
            pytest.param(f"{MODEL_CATALOG_API_PATH}models", id="model_catalog_models"),
            pytest.param(MODEL_CATALOG_API_PATH, id="model_catalog_root"),
        ],
    )
    def test_unauthenticated_access_denied(
        self: Self,
        model_catalog_routes: list[Route],
        api_path: str,
    ):
        """Verify that requests without an Authorization header are rejected."""
        url = f"https://{model_catalog_routes[0].instance.spec.host}:443{api_path}"
        response = requests.get(
            url=url,
            headers={},
            verify=False,
            timeout=60,
        )
        LOGGER.info(f"Unauthenticated response to {api_path}: {response.status_code}")
        assert response.status_code == 401
