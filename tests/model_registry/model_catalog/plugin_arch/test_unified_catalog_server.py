import pytest
import requests
import structlog
from ocp_resources.pod import Pod

from tests.model_registry.model_catalog.constants import CATALOG_CONTAINER

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
        assert "Catalog plugin becoming leader" in log, "Catalog plugin load did not complete successfully"

        LOGGER.info("Catalog pod initialized plugins successfully")

    @pytest.mark.parametrize(
        "endpoint, expected_status, expect_plugins",
        [
            pytest.param("healthz", "ok", False, id="test_healthz"),
            pytest.param("readyz", "ready", True, id="test_readyz"),
        ],
    )
    def test_health_endpoint(
        self,
        catalog_base_url: str,
        model_registry_rest_headers: dict[str, str],
        endpoint: str,
        expected_status: str,
        expect_plugins: bool,
    ):
        """Given the catalog server is running with registered plugins
        When querying a health endpoint
        Then the server reports the expected status
        """
        response = requests.get(
            f"{catalog_base_url}/{endpoint}", headers=model_registry_rest_headers, verify=False, timeout=30
        )
        assert response.ok, f"/{endpoint} returned {response.status_code}"

        body = response.json()
        assert body["status"] == expected_status, f"/{endpoint} check failed: {body}"

        if expect_plugins:
            assert "plugins" in body, f"/{endpoint} missing 'plugins' aggregation: {body}"
            unhealthy_plugins = {name: status for name, status in body["plugins"].items() if not status}
            assert not unhealthy_plugins, f"Unhealthy plugins detected: {unhealthy_plugins}"

        LOGGER.info(f"/{endpoint} returned: {body}")
