import json

import pytest

from tests.ai_safety.evalhub.mcp.constants import (
    EVALHUB_MCP_PROMPTS,
    EVALHUB_MCP_SERVER_NAME,
    EVALHUB_MCP_TOOLS,
)
from tests.ai_safety.evalhub.mcp.utils import (
    EvalHubMcpClient,
    McpProtocolError,
    mcp_prompt_names,
    mcp_read_resource_text,
    mcp_resource_names,
    mcp_tool_error_text,
    mcp_tool_names,
    read_mcp_resource,
)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-mcp-protocol"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.ai_safety
class TestEvalHubMcpProtocol:
    """MCP protocol tests for evalhub-mcp over Streamable HTTP (tools, resources, prompts)."""

    def test_initialize_advertises_server_metadata(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """
        Given: Authenticated MCP client
        When: MCP initialize request is sent
        Then: Response advertises evalhub-mcp server info and tools, resources, prompts capabilities
        """
        result = evalhub_mcp_client.initialize_result
        server_info = result.get("serverInfo", {})
        assert server_info.get("name") == EVALHUB_MCP_SERVER_NAME
        assert server_info.get("version"), "Expected non-empty server version"

        capabilities = result.get("capabilities", {})
        assert capabilities.get("tools") is not None, "Expected tools capability"
        assert capabilities.get("resources") is not None, "Expected resources capability"
        assert capabilities.get("prompts") is not None, "Expected prompts capability"

    def test_list_tools_includes_evalhub_tools(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """
        Given: Authenticated MCP client
        When: tools/list request is sent
        Then: Response includes all four EvalHub MCP tools
        """
        result = evalhub_mcp_client.call(method="tools/list", params={})
        tool_names = set(mcp_tool_names(result=result))
        for expected in EVALHUB_MCP_TOOLS:
            assert expected in tool_names, f"Expected tool '{expected}' in {tool_names}"

    def test_list_resources_includes_evalhub_resources(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """
        Given: Authenticated MCP client
        When: resources/list request is sent
        Then: Response includes providers, benchmarks, collections, jobs, and server version
        """
        result = evalhub_mcp_client.call(method="resources/list", params={})
        resource_names = set(mcp_resource_names(result=result))
        expected_names = {"providers", "benchmarks", "collections", "jobs", "server-version"}
        assert expected_names.issubset(resource_names), f"Expected resources {expected_names}, got {resource_names}"

    @pytest.mark.parametrize(
        "uri",
        [
            pytest.param("evalhub://server/../../etc/passwd", id="test_uri_traversal_dotdot"),
            pytest.param("evalhub://../../../admin/secrets", id="test_uri_traversal_root_escape"),
            pytest.param(
                "evalhub://server/%2e%2e/%2e%2e/etc/passwd",
                id="test_uri_traversal_encoded_dotdot",
            ),
        ],
    )
    def test_read_resource_rejects_traversal(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
        uri: str,
    ) -> None:
        """
        Given: A resource URI containing path traversal sequences
        When: resources/read is called with that URI
        Then: McpProtocolError is raised
        """
        with pytest.raises(McpProtocolError):
            read_mcp_resource(client=evalhub_mcp_client, uri=uri)

    def test_read_server_version_resource(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """
        Given: Authenticated MCP client
        When: resources/read is called for evalhub://server/version
        Then: Response returns JSON build metadata with version and mcp_library
        """
        result = evalhub_mcp_client.call(
            method="resources/read",
            params={"uri": "evalhub://server/version"},
        )
        text = mcp_read_resource_text(result=result)
        payload = json.loads(text)
        assert payload.get("version"), f"Expected version in server metadata: {payload}"
        assert payload.get("mcp_library"), "Expected mcp_library field in version resource"

    def test_read_providers_resource(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """
        Given: Authenticated MCP client
        When: resources/read is called for evalhub://providers
        Then: Response returns a JSON provider list
        """
        result = evalhub_mcp_client.call(
            method="resources/read",
            params={"uri": "evalhub://providers"},
        )
        text = mcp_read_resource_text(result=result)
        providers = json.loads(text)
        assert isinstance(providers, list), f"Expected list of providers, got: {type(providers)}"

    def test_read_benchmarks_resource(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """
        Given: Authenticated MCP client
        When: resources/read is called for evalhub://benchmarks
        Then: Response returns benchmark metadata as a JSON list
        """
        result = evalhub_mcp_client.call(
            method="resources/read",
            params={"uri": "evalhub://benchmarks"},
        )
        text = mcp_read_resource_text(result=result)
        benchmarks = json.loads(text)
        assert isinstance(benchmarks, list), f"Expected list of benchmarks, got: {type(benchmarks)}"

    def test_read_collections_resource(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """
        Given: Authenticated MCP client
        When: resources/read is called for evalhub://collections
        Then: Response returns collection metadata as a JSON list
        """
        result = evalhub_mcp_client.call(
            method="resources/read",
            params={"uri": "evalhub://collections"},
        )
        text = mcp_read_resource_text(result=result)
        collections = json.loads(text)
        assert isinstance(collections, list), f"Expected list of collections, got: {type(collections)}"

    def test_read_jobs_resource(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """
        Given: Authenticated MCP client
        When: resources/read is called for evalhub://jobs
        Then: Response returns a JSON job list
        """
        result = evalhub_mcp_client.call(
            method="resources/read",
            params={"uri": "evalhub://jobs"},
        )
        text = mcp_read_resource_text(result=result)
        jobs = json.loads(text)
        assert isinstance(jobs, list), f"Expected list of jobs, got: {type(jobs)}"

    def test_list_prompts_includes_workflow_prompts(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """
        Given: Authenticated MCP client
        When: prompts/list request is sent
        Then: Response includes edd_workflow, evaluate_model, and compare_runs
        """
        result = evalhub_mcp_client.call(method="prompts/list", params={})
        prompt_names = set(mcp_prompt_names(result=result))
        for expected in EVALHUB_MCP_PROMPTS:
            assert expected in prompt_names, f"Expected prompt '{expected}' in {prompt_names}"

    def test_get_edd_workflow_prompt_for_rag(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """
        Given: Authenticated MCP client with application_type rag
        When: prompts/get edd_workflow is called
        Then: Response includes user and assistant dialogue messages
        """
        result = evalhub_mcp_client.call(
            method="prompts/get",
            params={
                "name": "edd_workflow",
                "arguments": {"application_type": "rag"},
            },
        )
        messages = result.get("messages", [])
        assert messages, "Expected non-empty prompt messages"
        roles = {message.get("role") for message in messages if isinstance(message, dict)}
        assert "user" in roles or "assistant" in roles, f"Expected dialogue roles in {roles}"

    def test_get_evaluate_model_prompt_without_model_url(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """
        Given: Authenticated MCP client without model_url
        When: prompts/get evaluate_model is called
        Then: Response includes no_model message group
        """
        result = evalhub_mcp_client.call(
            method="prompts/get",
            params={"name": "evaluate_model", "arguments": {}},
        )
        assert result.get("messages"), "Expected messages for evaluate_model prompt"

    def test_completion_for_benchmark_id(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """
        Given: Authenticated MCP client with benchmark resource reference
        When: completion/complete is called for benchmark id
        Then: Response includes completion value suggestions
        """
        result = evalhub_mcp_client.call(
            method="completion/complete",
            params={
                "ref": {"type": "ref/resource", "uri": "evalhub://benchmarks/{id}"},
                "argument": {"name": "id", "value": ""},
            },
        )
        completion = result.get("completion", {})
        assert "values" in completion, f"Expected completion values, got: {result}"

    def test_discover_providers_tool_returns_providers(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """
        Given: Authenticated MCP client
        When: tools/call discover_providers is invoked
        Then: Response includes a provider summary list
        """
        result = evalhub_mcp_client.call(
            method="tools/call",
            params={
                "name": "discover_providers",
                "arguments": {},
            },
        )
        structured = result.get("structuredContent", {})
        providers = structured.get("providers", [])
        assert isinstance(providers, list), f"Expected providers list in structuredContent: {structured}"

    def test_get_job_status_requires_job_id(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """
        Given: Authenticated MCP client
        When: tools/call get_job_status is invoked without job_id
        Then: Tool returns a validation error mentioning job_id
        """
        result = evalhub_mcp_client.call(
            method="tools/call",
            params={"name": "get_job_status", "arguments": {}},
        )
        assert result.get("isError") is True
        text = mcp_tool_error_text(result=result)
        assert "job_id" in text.lower()

    def test_cancel_job_requires_job_id(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """
        Given: Authenticated MCP client
        When: tools/call cancel_job is invoked without job_id
        Then: Tool returns a validation error mentioning job_id
        """
        result = evalhub_mcp_client.call(
            method="tools/call",
            params={"name": "cancel_job", "arguments": {}},
        )
        assert result.get("isError") is True
        text = mcp_tool_error_text(result=result)
        assert "job_id" in text.lower()

    def test_get_job_status_nonexistent_job(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """
        Given: Authenticated MCP client with a nonexistent job ID
        When: tools/call get_job_status is invoked
        Then: Tool returns an error result
        """
        result = evalhub_mcp_client.call(
            method="tools/call",
            params={
                "name": "get_job_status",
                "arguments": {"job_id": "00000000-0000-0000-0000-000000000000"},
            },
        )
        assert result.get("isError") is True
