import uuid

import pytest
from ocp_resources.namespace import Namespace
from ocp_resources.service import Service

from tests.ai_safety.evalhub.mcp.utils import (
    EvalHubMcpClient,
    build_mcp_evaluation_arguments,
    build_mcp_model_url,
    call_mcp_tool,
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
@pytest.mark.usefixtures(
    "evalhub_mcp_vllm_emulator_deployment",
    "evalhub_mcp_vllm_emulator_service",
    "tenant_a_mcp_rbac_ready",
)
class TestEvalHubMcpJobs:
    """MCP job lifecycle tests: status polling, cancellation, and completion."""

    def test_get_job_status_returns_progress_fields(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
        evalhub_mcp_vllm_emulator_service: Service,
        tenant_a_namespace: Namespace,
    ) -> None:
        """get_job_status returns state and progress_percent for a submitted job."""
        model_url = build_mcp_model_url(
            service_name=evalhub_mcp_vllm_emulator_service.name,
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
        evalhub_mcp_vllm_emulator_service: Service,
        tenant_a_namespace: Namespace,
    ) -> None:
        """cancel_job stops a running evaluation and get_job_status reports cancelled."""
        model_url = build_mcp_model_url(
            service_name=evalhub_mcp_vllm_emulator_service.name,
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

        terminal_state = wait_for_mcp_job_state(
            client=evalhub_mcp_client,
            job_id=job_id,
            timeout=300,
            terminal_states={"cancelled", "failed", "completed", "partially_failed"},
        )
        assert terminal_state == "cancelled", f"Expected cancelled state, got '{terminal_state}'"

    def test_cancel_nonexistent_job_returns_error(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """cancel_job for a missing job ID returns a tool error."""
        result = call_mcp_tool(
            client=evalhub_mcp_client,
            name="cancel_job",
            arguments={"job_id": "00000000-0000-0000-0000-000000000000"},
        )
        assert mcp_tool_is_error(result=result), f"Expected error cancelling missing job, got: {result}"
