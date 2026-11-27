from __future__ import annotations

from collections.abc import Callable

import portforward
import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace

from tests.ai_hub.mcp_servers.integration.constants import (
    DEPLOYABLE_MCP_EXPECTED_TOOL_NAMES,
    DEPLOYABLE_MCP_PATH,
    DEPLOYABLE_MCP_VERSION,
    REGISTRY_SERVER_ANNOTATION,
    REGISTRY_VERSION_ANNOTATION,
)
from tests.ai_hub.mcp_servers.integration.utils import probe_mcp_tools
from utilities.resources.mcp_server import MCPServer


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-mcp-registry-happy-path"},
            marks=[pytest.mark.install, pytest.mark.tier2],
            id="test_mcp_registry_backend_happy_path",
        )
    ],
    indirect=True,
)
def test_register_and_deploy_mcp_server_from_catalog_backend_happy_path(
    admin_client: DynamicClient,
    mcp_access_endpoint: dict[str, object],
    model_namespace: Namespace,
    unused_tcp_port_factory: Callable[[], int],
) -> None:
    """
    Given: A deployable MCP server exists in the live AI Hub MCP catalog
    When: Its catalog metadata is registered in MLflow and the registered version is deployed through the
          dashboard BFF APIs
    Then: The deployment becomes Ready, serves MCP tools over HTTP, and the backing MCPServer CR is
          linked to the registry version
    """
    workspace_name = model_namespace.name
    deployment_name = str(mcp_access_endpoint["deployment_name"])
    ready_deployment = dict(mcp_access_endpoint["ready_deployment"])
    server_name = str(mcp_access_endpoint["server_name"])

    mcp_server = MCPServer(
        client=admin_client,
        name=deployment_name,
        namespace=workspace_name,
        ensure_exists=True,
    )
    annotations = mcp_server.instance.metadata.annotations or {}
    assert annotations.get(REGISTRY_SERVER_ANNOTATION) == server_name
    assert annotations.get(REGISTRY_VERSION_ANNOTATION) == DEPLOYABLE_MCP_VERSION

    local_probe_port = unused_tcp_port_factory()
    deployment_port = int(ready_deployment["port"])
    deployment_path = ready_deployment.get("path") or DEPLOYABLE_MCP_PATH
    with portforward.forward(
        pod_or_service=deployment_name,
        namespace=workspace_name,
        from_port=local_probe_port,
        to_port=deployment_port,
        waiting=20,
    ):
        deployed_tools = probe_mcp_tools(endpoint_url=f"http://127.0.0.1:{local_probe_port}{deployment_path}")

    deployed_tool_names = {tool.name for tool in deployed_tools}
    assert deployed_tool_names, f"Expected deployed MCP server {deployment_name} to advertise tools"
    assert set(DEPLOYABLE_MCP_EXPECTED_TOOL_NAMES).issubset(deployed_tool_names), (
        f"Expected deployed MCP server {deployment_name} to expose "
        f"{sorted(DEPLOYABLE_MCP_EXPECTED_TOOL_NAMES)}, got: "
        f"{sorted(deployed_tool_names)}"
    )
