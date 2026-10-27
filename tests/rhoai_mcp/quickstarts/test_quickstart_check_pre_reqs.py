"""Quickstart CHECK_PRE_REQS action tests for rhoai-mcp.

Runs the known quickstart's pre-requisite check through the MCP server, which
launches the quickstart's installer image as a Kubernetes Job, then verifies the
Job can be tracked to completion and its logs read back through MCP tools.
"""

from typing import Any

import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from tests.rhoai_mcp.quickstarts.constants import (
    RHOAI_MCP_QUICKSTART_CHECK_PRE_REQS_ACTION,
    RHOAI_MCP_QUICKSTART_JOB_NAMESPACE,
    RHOAI_MCP_QUICKSTART_LOGS_UNAVAILABLE,
    RHOAI_MCP_QUICKSTART_NAME,
    RHOAI_MCP_QUICKSTART_PRE_REQS_OUTCOMES,
)
from tests.rhoai_mcp.quickstarts.utils import wait_for_quickstart_action_finished
from tests.rhoai_mcp.utils import parse_tool_result


@pytest.mark.asyncio
@pytest.mark.tier1
class TestRhoaiMcpQuickstartCheckPreReqs:
    """Validate the CHECK_PRE_REQS action lifecycle using rhoai-mcp MCP tools.

    Steps:
        1. Run CHECK_PRE_REQS via run_quickstart_action (creates an installer Job).
        2. Poll get_quickstart_action_status until the Job finishes and reports the installer's verdict.
        3. Read the installer output via get_quickstart_action_logs.
    """

    @pytest.mark.dependency(name="pre_reqs_job_created")
    async def test_check_pre_reqs_creates_installer_job(
        self,
        check_pre_reqs_action: dict[str, Any],
    ) -> None:
        """Given rhoai-mcp is deployed and the quickstart installer identity is provisioned
        When run_quickstart_action is called with the CHECK_PRE_REQS action
        Then the server creates an installer Job in the quickstart job namespace
        """
        assert "error" not in check_pre_reqs_action, f"run_quickstart_action failed: {check_pre_reqs_action}"
        assert check_pre_reqs_action["quickstart"] == RHOAI_MCP_QUICKSTART_NAME
        assert check_pre_reqs_action["action"] == RHOAI_MCP_QUICKSTART_CHECK_PRE_REQS_ACTION
        assert check_pre_reqs_action["job_namespace"] == RHOAI_MCP_QUICKSTART_JOB_NAMESPACE
        assert check_pre_reqs_action["job_name"], "run_quickstart_action returned no job name"

    @pytest.mark.dependency(name="pre_reqs_job_finished", depends=["pre_reqs_job_created"])
    async def test_check_pre_reqs_job_reports_result(
        self,
        rhoai_mcp_transport: StreamableHttpTransport,
        check_pre_reqs_action: dict[str, Any],
    ) -> None:
        """Given a CHECK_PRE_REQS installer Job has been created
        When get_quickstart_action_status is polled until the Job finishes
        Then the status reports the installer's verdict, with a phase that matches it

        Note: the verdict itself depends on the cluster, so both outcomes are accepted.
        This verifies that rhoai-mcp tracks the Job and reports the installer's result.
        """
        async with Client(rhoai_mcp_transport) as client:
            status = await wait_for_quickstart_action_finished(
                client=client,
                job_name=check_pre_reqs_action["job_name"],
            )

        assert status["phase"] in RHOAI_MCP_QUICKSTART_PRE_REQS_OUTCOMES, f"Unexpected Job phase: {status}"
        result = status["result"] or {}
        assert result.get("status") == RHOAI_MCP_QUICKSTART_PRE_REQS_OUTCOMES[status["phase"]], (
            f"CHECK_PRE_REQS Job did not report a result matching its phase: {status}"
        )
        assert status["quickstart"] == RHOAI_MCP_QUICKSTART_NAME
        assert status["action"] == RHOAI_MCP_QUICKSTART_CHECK_PRE_REQS_ACTION

    @pytest.mark.dependency(depends=["pre_reqs_job_finished"])
    async def test_check_pre_reqs_logs_available(
        self,
        rhoai_mcp_transport: StreamableHttpTransport,
        check_pre_reqs_action: dict[str, Any],
    ) -> None:
        """Given a CHECK_PRE_REQS installer Job has finished
        When get_quickstart_action_logs is called for that Job
        Then the installer's log output is returned
        """
        async with Client(rhoai_mcp_transport) as client:
            result = await client.call_tool(
                name="get_quickstart_action_logs",
                arguments={"job_name": check_pre_reqs_action["job_name"]},
            )
            data = parse_tool_result(result=result)

        assert "error" not in data, f"get_quickstart_action_logs failed: {data}"
        assert data["logs"].strip(), "Installer logs are empty"
        assert data["logs"] != RHOAI_MCP_QUICKSTART_LOGS_UNAVAILABLE, "Installer logs could not be read"
