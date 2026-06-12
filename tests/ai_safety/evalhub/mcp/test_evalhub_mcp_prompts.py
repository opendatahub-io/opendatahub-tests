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
        """
        Given: Authenticated MCP client with a valid application_type
        When: prompts/get is called for edd_workflow
        Then: Response includes non-empty dialogue messages
        """
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
        """
        Given: Authenticated MCP client
        When: prompts/get edd_workflow is called with invalid application_type
        Then: Server returns an MCP protocol error
        """
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
        """
        Given: Authenticated MCP client with a model_url argument
        When: prompts/get evaluate_model is called
        Then: Response includes user or assistant dialogue messages
        """
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
        """
        Given: Authenticated MCP client without job_ids
        When: prompts/get compare_runs is called
        Then: Response includes no_jobs guidance messages
        """
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
        """
        Given: Authenticated MCP client with two job IDs
        When: prompts/get compare_runs is called
        Then: Response includes comparison guidance messages
        """
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
        """
        Given: Authenticated MCP client with a single job ID
        When: prompts/get compare_runs is called
        Then: Server returns an MCP protocol error
        """
        with pytest.raises(McpProtocolError):
            get_mcp_prompt(
                client=evalhub_mcp_client,
                name="compare_runs",
                arguments={"job_ids": "only-one-job"},
            )
