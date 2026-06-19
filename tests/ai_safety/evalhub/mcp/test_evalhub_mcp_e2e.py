import pytest
from ocp_resources.namespace import Namespace
from ocp_resources.service import Service

from tests.ai_safety.evalhub.mcp.utils import (
    EvalHubMcpClient,
    build_mcp_evaluation_arguments,
    build_mcp_model_url,
    call_mcp_tool,
    format_mcp_job_status_failure,
    mcp_tool_structured,
    wait_for_mcp_job_state,
)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-mcp-e2e"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier2
@pytest.mark.ai_safety
class TestEvalHubMcpE2E:
    """End-to-end MCP workflow: discover providers, submit job, poll status."""

    def test_submit_and_monitor_evaluation_via_mcp_tools(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
        tenant_a_namespace: Namespace,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """
        Given: Authenticated MCP client and a vLLM emulator model endpoint
        When: discover_providers, submit_evaluation, and get_job_status are called
        Then: Job is created and reaches completed state
        """
        model_url = build_mcp_model_url(
            service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
        )

        discover_result = call_mcp_tool(
            client=evalhub_mcp_client,
            name="discover_providers",
            arguments={},
        )
        providers = mcp_tool_structured(result=discover_result).get("providers", [])
        provider_ids = {provider.get("id") for provider in providers if isinstance(provider, dict)}
        assert "lm_evaluation_harness" in provider_ids, f"Expected lm_evaluation_harness provider, got: {provider_ids}"

        submit_result = call_mcp_tool(
            client=evalhub_mcp_client,
            name="submit_evaluation",
            arguments=build_mcp_evaluation_arguments(
                model_url=model_url,
                job_name="mcp-e2e-eval-job",
            ),
        )
        assert submit_result.get("isError") is not True, f"Job submission failed: {submit_result}"
        structured = mcp_tool_structured(result=submit_result)
        job_id = structured.get("job_id")
        assert job_id, f"Expected job_id in submit response: {submit_result}"

        terminal_state, job_status = wait_for_mcp_job_state(client=evalhub_mcp_client, job_id=job_id)
        assert terminal_state == "completed", (
            f"Expected job to complete, got {format_mcp_job_status_failure(status=job_status)}"
        )
