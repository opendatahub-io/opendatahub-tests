from collections.abc import Generator
from typing import Any

import httpx
import pytest
import requests
from kubernetes.dynamic import DynamicClient
from ogx_client import OgxClient

from tests.ogx.constants import OGX_CLIENT_VERIFY_SSL
from tests.ogx.praxis.constants import (
    CONVERSATIONS_API_PATH,
    FILES_API_PATH,
    REQUEST_TIMEOUT_SECONDS,
    RESPONSES_API_PATH,
    VECTOR_STORES_API_PATH,
)
from tests.ogx.praxis.utils import (
    gateway_base_url,
    praxis_api_url,
    praxis_gateway_base_url,
    praxis_http_route,
)
from utilities.infra import get_openshift_token
from utilities.resources.http_route import HTTPRoute


@pytest.fixture
def tenant_authorization_header(admin_client: DynamicClient) -> dict[str, str]:
    """Authorization header carrying the OpenShift token of the authenticated tenant."""
    return {
        "Authorization": f"Bearer {get_openshift_token(client=admin_client)}",
        "Content-Type": "application/json",
    }


@pytest.fixture
def request_session() -> Generator[requests.Session, Any, Any]:
    """HTTP session for requests that must bypass the OGX SDK and its authentication."""
    session = requests.Session()
    session.verify = OGX_CLIENT_VERIFY_SSL
    yield session
    session.close()


@pytest.fixture(scope="class")
def praxis_client(admin_client: DynamicClient) -> Generator[OgxClient, Any, Any]:
    """OgxClient bound to the external Gateway hostname that Praxis serves.

    The shared `ogx_client` addresses the OGX Service directly through an OpenShift
    Route, so it reaches OGX whatever the external routing does. Assertions about
    Praxis-era behaviour must traverse the Gateway -> Praxis -> OGX path instead.
    """
    http_client = httpx.Client(verify=OGX_CLIENT_VERIFY_SSL, timeout=REQUEST_TIMEOUT_SECONDS)
    try:
        yield OgxClient(
            base_url=praxis_gateway_base_url(
                client=admin_client,
                paths=(RESPONSES_API_PATH, FILES_API_PATH, VECTOR_STORES_API_PATH, CONVERSATIONS_API_PATH),
            ),
            api_key=get_openshift_token(client=admin_client),
            http_client=http_client,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )
    finally:
        http_client.close()


@pytest.fixture(scope="class")
def praxis_responses_http_route(admin_client: DynamicClient) -> HTTPRoute:
    """HTTPRoute publishing the Responses API at the public Gateway boundary."""
    return praxis_http_route(client=admin_client, path=RESPONSES_API_PATH)


@pytest.fixture(scope="class")
def praxis_files_http_route(admin_client: DynamicClient) -> HTTPRoute:
    """HTTPRoute publishing the Files API at the public Gateway boundary."""
    return praxis_http_route(client=admin_client, path=FILES_API_PATH)


@pytest.fixture(scope="class")
def praxis_responses_url(praxis_responses_http_route: HTTPRoute) -> str:
    """Public URL of the Responses API at the Praxis Gateway boundary."""
    return f"{gateway_base_url(http_route=praxis_responses_http_route)}{RESPONSES_API_PATH}"


@pytest.fixture(scope="class")
def praxis_files_url(praxis_files_http_route: HTTPRoute) -> str:
    """Public URL of the Files API at the Praxis Gateway boundary."""
    return f"{gateway_base_url(http_route=praxis_files_http_route)}{FILES_API_PATH}"


@pytest.fixture(scope="class")
def praxis_vector_stores_url(admin_client: DynamicClient) -> str:
    """Public URL of the Vector Stores API at the Praxis Gateway boundary."""
    return praxis_api_url(client=admin_client, path=VECTOR_STORES_API_PATH)


@pytest.fixture(scope="class")
def praxis_conversations_url(admin_client: DynamicClient) -> str:
    """Public URL of the Conversations API at the Praxis Gateway boundary."""
    return praxis_api_url(client=admin_client, path=CONVERSATIONS_API_PATH)
