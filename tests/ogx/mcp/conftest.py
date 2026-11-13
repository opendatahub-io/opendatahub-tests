"""Fixtures for the MCP tool runtime tests.

The ``ogx_server`` fixture defined here shadows the one in
``tests/ogx/conftest.py`` for tests under ``tests/ogx/mcp/``. The server is
deployed with a run.yaml override that:

- registers an in-cluster mock MCP server as a connector (startup
  registration; exposed read-only via the admin connectors API), and
- enables the responses API with the ``inline::builtin`` provider, which is
  what executes MCP tools server-side on behalf of a model.

The model itself is the mock MaaS upstream through the
``genai-bff-proxy`` ``remote::passthrough`` provider: it deterministically
requests the mock MCP server's ``get_weather`` tool, so the whole loop
(model tool-call -> OGX MCP invocation -> tool result in the response) is
assertable without any external dependency.
"""

from collections.abc import Generator
from typing import Any

import pytest
import yaml
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.namespace import Namespace

from tests.ogx.mcp.constants import (
    MCP_CONFIGMAP_KEY,
    MCP_CONFIGMAP_NAME,
    MCP_SERVER_NAME,
)
from tests.ogx.mockupstream.constants import MOCK_INFERENCE_API_KEY
from tests.ogx.mockupstream.deployments import MockUpstream, deploy_mock_mcp_server
from tests.ogx.mockupstream.ogx_config import (
    build_mock_run_config,
    build_override_config_server_spec,
)
from tests.ogx.utils import create_ogx_server
from utilities.resources.ogx_server import OgxServer


@pytest.fixture(scope="class")
def mock_mcp_upstream(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[MockUpstream, Any, Any]:
    """Deploy the mock MCP server in the test namespace (in-cluster only)."""
    with deploy_mock_mcp_server(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace,
        teardown=teardown_resources,
    ) as upstream:
        yield upstream


@pytest.fixture(scope="class")
def ogx_server(  # noqa: UFN001 — intentionally shadows the parent conftest fixture (see module docstring)
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    enabled_ogx_operator: DataScienceCluster,
    mock_mcp_upstream: MockUpstream,
    mock_maas_upstream: MockUpstream,
    teardown_resources: bool,
    request: FixtureRequest,
) -> Generator[OgxServer, Any, Any]:
    """OGXServer with the MCP connector and the responses API enabled."""
    namespace = unprivileged_model_namespace.name
    config_yaml = yaml.dump(
        build_mock_run_config(
            distro_name=MCP_SERVER_NAME,
            mock_service_url=mock_maas_upstream.service_url,
            api_key=MOCK_INFERENCE_API_KEY,
            mcp_endpoint=mock_mcp_upstream.service_url,
        )
    )

    if pytestconfig.option.post_upgrade:
        ogx_srv = OgxServer(
            client=unprivileged_client,
            name=MCP_SERVER_NAME,
            namespace=namespace,
            ensure_exists=True,
        )
        ogx_srv.wait_for_status(status=OgxServer.Status.READY, timeout=600)
        yield ogx_srv
        if teardown_resources:
            ogx_srv.clean_up()
        return

    with (
        ConfigMap(
            client=unprivileged_client,
            name=MCP_CONFIGMAP_NAME,
            namespace=namespace,
            data={MCP_CONFIGMAP_KEY: config_yaml},
            label={"ogx.io/watch": "true"},
            wait_for_resource=True,
            teardown=teardown_resources,
        ),
        create_ogx_server(
            client=unprivileged_client,
            name=MCP_SERVER_NAME,
            namespace=namespace,
            config=build_override_config_server_spec(
                configmap_name=MCP_CONFIGMAP_NAME,
                configmap_key=MCP_CONFIGMAP_KEY,
            ),
            teardown=teardown_resources,
        ) as ogx_srv,
    ):
        ogx_srv.wait_for_status(status=OgxServer.Status.READY, timeout=600)
        yield ogx_srv


@pytest.fixture(scope="class")
def ogx_admin_api_url(ogx_test_route) -> str:
    """Base URL of the OGX server, for admin APIs not covered by the OgxClient."""
    return f"https://{ogx_test_route.host}"
