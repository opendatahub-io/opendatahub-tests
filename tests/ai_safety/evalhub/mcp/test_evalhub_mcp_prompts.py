import pytest

from tests.ai_safety.evalhub.mcp.constants import EVALHUB_MCP_EDD_APPLICATION_TYPES
from tests.ai_safety.evalhub.mcp.utils import (
    EvalHubMcpClient,
    McpProtocolError,
    get_mcp_prompt,
)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-mcp-prompts"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.ai_safety
class TestEvalHubMcpPrompts:
    """MCP workflow prompt tests (edd_workflow, evaluate_model, compare_runs)."""

    @pytest.mark.parametrize("application_type", EVALHUB_MCP_EDD_APPLICATION_TYPES)
    def test_get_edd_workflow_prompt_for_application_type(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
        application_type: str,
    ) -> None:
        """prompts/get edd_workflow returns dialogue for each valid application_type."""
        result = get_mcp_prompt(
            client=evalhub_mcp_client,
            name="edd_workflow",
            arguments={"application_type": application_type},
        )
        messages = result.get("messages", [])
        assert messages, f"Expected messages for application_type={application_type}"

    def test_get_edd_workflow_invalid_application_type(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """prompts/get edd_workflow with invalid application_type returns an error."""
        with pytest.raises(McpProtocolError):
            get_mcp_prompt(
                client=evalhub_mcp_client,
                name="edd_workflow",
                arguments={"application_type": "invalid-type"},
            )

    def test_get_evaluate_model_prompt_with_model_url(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """prompts/get evaluate_model with model_url uses the with_model message group."""
        result = get_mcp_prompt(
            client=evalhub_mcp_client,
            name="evaluate_model",
            arguments={"model_url": "http://model.example/v1"},
        )
        messages = result.get("messages", [])
        assert messages
        roles = {message.get("role") for message in messages if isinstance(message, dict)}
        assert "user" in roles or "assistant" in roles

    def test_get_compare_runs_prompt_without_job_ids(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """prompts/get compare_runs without job_ids returns the no_jobs guidance."""
        result = get_mcp_prompt(
            client=evalhub_mcp_client,
            name="compare_runs",
            arguments={},
        )
        messages = result.get("messages", [])
        assert messages, "Expected compare_runs messages for empty job_ids"

    def test_get_compare_runs_prompt_with_two_job_ids(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """prompts/get compare_runs with two job IDs returns comparison guidance."""
        result = get_mcp_prompt(
            client=evalhub_mcp_client,
            name="compare_runs",
            arguments={"job_ids": "job-a, job-b"},
        )
        messages = result.get("messages", [])
        assert len(messages) >= 2, "Expected compare_runs dialogue plus comparison guidance"

    def test_get_compare_runs_prompt_with_single_job_id_fails(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """prompts/get compare_runs with one job ID requires at least two IDs."""
        with pytest.raises(McpProtocolError):
            get_mcp_prompt(
                client=evalhub_mcp_client,
                name="compare_runs",
                arguments={"job_ids": "only-one-job"},
            )
