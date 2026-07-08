from collections.abc import Generator
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.route import Route

from tests.ai_hub.plugin_arch.utils import (
    READYZ_RECOVERY_TIMEOUT,
    poll_readyz,
    restore_catalog,
)

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="class")
def catalog_base_url(model_catalog_routes: list[Route]) -> str:
    """Base URL for the catalog server without the API path."""
    assert model_catalog_routes, "Model catalog routes not found"
    return f"https://{model_catalog_routes[0].instance.spec.host}:443"


@pytest.fixture()
def healthy_catalog_state(
    admin_client: DynamicClient,
    catalog_base_url: str,
    model_registry_rest_headers: dict[str, str],
    model_registry_namespace: str,
) -> Generator[None, Any, Any]:
    """Ensure catalog DB login is restored, stale locks are cleared, and /readyz is 200."""
    yield

    LOGGER.info("Restoring cluster state after test")
    restore_catalog(admin_client=admin_client, namespace=model_registry_namespace)
    poll_readyz(
        url=f"{catalog_base_url}/readyz",
        headers=model_registry_rest_headers,
        expected_code=200,
        timeout=READYZ_RECOVERY_TIMEOUT,
    )
