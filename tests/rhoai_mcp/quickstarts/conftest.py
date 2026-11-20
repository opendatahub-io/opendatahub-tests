from collections.abc import Generator
from typing import Any

import pytest
from fastmcp.client.transports import StreamableHttpTransport
from kubernetes.dynamic import DynamicClient
from ocp_resources.job import Job

from tests.rhoai_mcp.quickstarts.constants import (
    RHOAI_MCP_QUICKSTART_CHECK_PRE_REQS_ACTION,
    RHOAI_MCP_QUICKSTART_NAME,
)
from tests.rhoai_mcp.quickstarts.utils import call_mcp_tool


@pytest.fixture(scope="class")
def check_pre_reqs_action(
    admin_client: DynamicClient,
    rhoai_mcp_transport: StreamableHttpTransport,
) -> Generator[dict[str, Any], Any, Any]:
    """Response from a CHECK_PRE_REQS action run via MCP; its installer Job is deleted on teardown."""
    response = call_mcp_tool(
        transport=rhoai_mcp_transport,
        name="run_quickstart_action",
        arguments={"name": RHOAI_MCP_QUICKSTART_NAME, "action": RHOAI_MCP_QUICKSTART_CHECK_PRE_REQS_ACTION},
    )
    yield response
    if "job_name" in response:
        Job(
            client=admin_client,
            name=response["job_name"],
            namespace=response["job_namespace"],
            background_propagation_policy="Background",
        ).delete(wait=True)
