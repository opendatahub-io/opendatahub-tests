"""Fixtures local to serving runtime catalog integration coverage."""

from collections.abc import Generator

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.route import Route

from tests.ai_hub.serving_runtime_catalog.constants import RUNTIME_API_PATH, SOURCE_API_PATH, USER_CONFIGMAP
from tests.ai_hub.serving_runtime_catalog.utils import configured_runtime_sources
from tests.ai_hub.utils import get_rest_headers


@pytest.fixture(scope="package")
def runtime_catalog_routes(admin_client: DynamicClient, model_registry_namespace: str) -> list[Route]:
    """Model catalog routes, widened to package scope so the shared source install can use them."""
    return list(
        Route.get(namespace=model_registry_namespace, label_selector="component=model-catalog", client=admin_client)
    )


@pytest.fixture(scope="package")
def runtime_rest_headers(current_client_token: str) -> dict[str, str]:
    """Authentication headers, widened to package scope so the shared source install can use them."""
    return get_rest_headers(token=current_client_token)


@pytest.fixture(scope="package")
def runtime_base_url(runtime_catalog_routes: list[Route]) -> str:
    """Catalog server URL derived from the operator-managed route."""
    assert runtime_catalog_routes, "Model catalog route does not exist"
    return f"https://{runtime_catalog_routes[0].instance.spec.host}:443"


@pytest.fixture(scope="package")
def runtime_api_url(runtime_base_url: str) -> str:
    """Serving runtime catalog API base URL."""
    return f"{runtime_base_url}{RUNTIME_API_PATH}"


@pytest.fixture(scope="package")
def runtime_source_url(runtime_base_url: str) -> str:
    """Shared source administration API base URL."""
    return f"{runtime_base_url}{SOURCE_API_PATH}"


@pytest.fixture(scope="package")
def runtime_source_configmap(admin_client: DynamicClient, model_registry_namespace: str) -> ConfigMap:
    """Existing ownerless user source ConfigMap managed by the operator."""
    return ConfigMap(client=admin_client, namespace=model_registry_namespace, name=USER_CONFIGMAP, ensure_exists=True)


@pytest.fixture(scope="package")
def runtime_sources(
    runtime_source_configmap: ConfigMap,
    runtime_api_url: str,
    runtime_source_url: str,
    runtime_rest_headers: dict[str, str],
) -> Generator[None]:
    """Controlled test sources installed once for the package; mutating tests restore their own patches."""
    with configured_runtime_sources(
        configmap=runtime_source_configmap,
        url=runtime_api_url,
        source_url=runtime_source_url,
        headers=runtime_rest_headers,
    ):
        yield
