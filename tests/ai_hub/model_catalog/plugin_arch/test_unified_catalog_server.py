import pytest
import requests
import structlog
from ocp_resources.pod import Pod

from tests.ai_hub.constants import CATALOG_CONTAINER

LOGGER = structlog.get_logger(name=__name__)

pytestmark = [
    pytest.mark.tier1,
    pytest.mark.usefixtures(
        "updated_dsc_component_state_scope_session",
        "model_registry_namespace",
    ),
]


class TestCatalogPluginArchitecture:
    """Tests for unified catalog server plugin architecture."""

    def test_catalog_pod_initializes_plugins(
        self,
        model_catalog_pod: Pod,
    ):
        """Given the catalog server starts with the plugin architecture
        When checking pod logs for initialization
        Then plugin initialization messages confirm successful startup
        """
        log = model_catalog_pod.log(container=CATALOG_CONTAINER)
        assert "all plugins initialized" in log, "Plugin initialization did not complete successfully"
        assert "model plugin becoming leader" in log, "Model plugin did not become leader"
        assert "mcp plugin becoming leader" in log, "MCP plugin did not become leader"

        LOGGER.info("Catalog pod initialized all plugins successfully")

    def test_healthz(
        self,
        catalog_base_url: str,
        model_registry_rest_headers: dict[str, str],
    ):
        """Given the catalog server is running
        When querying the healthz endpoint
        Then the server reports ok status
        """
        response = requests.get(
            f"{catalog_base_url}/healthz", headers=model_registry_rest_headers, verify=False, timeout=30
        )
        assert response.ok, f"/healthz returned {response.status_code}"

        body = response.json()
        assert body["status"] == "ok", f"/healthz check failed: {body}"
        LOGGER.info(f"/healthz returned: {body}")

    @pytest.mark.parametrize(
        "plugin_name",
        [
            pytest.param("model", id="test_readyz_model_plugin"),
            pytest.param("mcp", id="test_readyz_mcp_plugin"),
        ],
    )
    def test_readyz_plugin_registered(
        self,
        catalog_base_url: str,
        model_registry_rest_headers: dict[str, str],
        plugin_name: str,
    ):
        """Given the catalog server is running with registered plugins
        When querying the readyz endpoint
        Then the plugin is registered and healthy
        """
        response = requests.get(
            f"{catalog_base_url}/readyz", headers=model_registry_rest_headers, verify=False, timeout=30
        )
        assert response.ok, f"/readyz returned {response.status_code}"

        body = response.json()
        assert body["status"] == "ready", f"/readyz check failed: {body}"
        assert body.get("plugins", {}).get(plugin_name) is True, (
            f"Plugin '{plugin_name}' not registered or unhealthy: {body}"
        )
        LOGGER.info(f"/readyz plugin '{plugin_name}': {body}")
