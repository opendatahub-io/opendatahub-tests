import uuid

import pytest
from ocp_resources.namespace import Namespace
from ocp_resources.service import Service

from tests.ai_safety.evalhub.mcp.utils import (
    EvalHubMcpClient,
    build_mcp_evaluation_arguments,
    build_mcp_model_url,
    call_mcp_tool,
    format_mcp_job_status_failure,
    mcp_tool_error_text,
    mcp_tool_is_error,
    mcp_tool_structured,
    submit_evaluation_via_mcp,
    wait_for_mcp_job_state,
)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-mcp-jobs"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier2
@pytest.mark.ai_safety
class TestEvalHubMcpJobs:
    """MCP job lifecycle tests: status polling, cancellation, and completion."""

    def test_get_job_status_returns_progress_fields(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
        evalhub_vllm_emulator_service: Service,
        tenant_a_namespace: Namespace,
    ) -> None:
        """
        Given: An evaluation job submitted via MCP tools
        When: get_job_status is called for the job ID
        Then: Response includes job_id, state, and progress_percent
        """
        model_url = build_mcp_model_url(
            service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
        )
        submit_result = submit_evaluation_via_mcp(
            client=evalhub_mcp_client,
            arguments=build_mcp_evaluation_arguments(
                model_url=model_url,
                job_name=f"mcp-status-fields-{uuid.uuid4().hex[:8]}",
            ),
        )
        job_id = submit_result["job_id"]

        status_result = call_mcp_tool(
            client=evalhub_mcp_client,
            name="get_job_status",
            arguments={"job_id": job_id},
        )
        assert not mcp_tool_is_error(result=status_result), mcp_tool_error_text(result=status_result)
        structured = mcp_tool_structured(result=status_result)
        assert structured.get("job_id") == job_id
        assert structured.get("state")
        assert "progress_percent" in structured

    def test_cancel_running_job(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
        evalhub_vllm_emulator_service: Service,
        tenant_a_namespace: Namespace,
    ) -> None:
        """
        Given: A running evaluation job submitted via MCP tools
        When: cancel_job is called and status is polled
        Then: Job reaches cancelled terminal state
        """
        model_url = build_mcp_model_url(
            service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
        )
        submit_result = submit_evaluation_via_mcp(
            client=evalhub_mcp_client,
            arguments=build_mcp_evaluation_arguments(
                model_url=model_url,
                job_name=f"mcp-cancel-job-{uuid.uuid4().hex[:8]}",
            ),
        )
        job_id = submit_result["job_id"]

        cancel_result = call_mcp_tool(
            client=evalhub_mcp_client,
            name="cancel_job",
            arguments={"job_id": job_id},
        )
        assert not mcp_tool_is_error(result=cancel_result), mcp_tool_error_text(result=cancel_result)
        structured_cancel = mcp_tool_structured(result=cancel_result)
        assert structured_cancel.get("job_id") == job_id

        terminal_state, job_status = wait_for_mcp_job_state(
            client=evalhub_mcp_client,
            job_id=job_id,
            timeout=300,
            terminal_states={"cancelled", "failed", "completed", "partially_failed"},
        )
        assert terminal_state == "cancelled", (
            f"Expected cancelled state, got {format_mcp_job_status_failure(status=job_status)}"
        )

    def test_cancel_nonexistent_job_returns_error(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """
        Given: Authenticated MCP client with no matching job
        When: cancel_job is called with a nonexistent job ID
        Then: Tool returns an error result
        """
        result = call_mcp_tool(
            client=evalhub_mcp_client,
            name="cancel_job",
            arguments={"job_id": "00000000-0000-0000-0000-000000000000"},
        )
        assert mcp_tool_is_error(result=result), f"Expected error cancelling missing job, got: {result}"
