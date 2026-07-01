import uuid

import pytest
from ocp_resources.namespace import Namespace
from ocp_resources.service import Service

from tests.ai_safety.evalhub.mcp.constants import (
    EVALHUB_MCP_DEFAULT_COLLECTION_ID,
    EVALHUB_MCP_DEFAULT_PROVIDER_ID,
)
from tests.ai_safety.evalhub.mcp.utils import (
    EvalHubMcpClient,
    build_mcp_evaluation_arguments,
    build_mcp_model_url,
    call_mcp_tool,
    mcp_tool_error_text,
    mcp_tool_is_error,
    mcp_tool_structured,
)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-mcp-tools"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.ai_safety
class TestEvalHubMcpTools:
    """MCP tool validation and discover_providers filtering."""

    def test_discover_providers_filter_by_target_type(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """
        Given: Authenticated MCP client
        When: discover_providers is called with target_type=model
        Then: Response returns only model-scoped providers
        """
        result = call_mcp_tool(
            client=evalhub_mcp_client,
            name="discover_providers",
            arguments={"target_type": "model"},
        )
        structured = mcp_tool_structured(result=result)
        providers = structured.get("providers", [])
        assert isinstance(providers, list)
        for provider in providers:
            if isinstance(provider, dict) and provider.get("target_type"):
                assert provider["target_type"] == "model"

    def test_discover_providers_returns_lm_evaluation_harness(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """
        Given: Authenticated MCP client
        When: discover_providers is called without filters
        Then: Response includes the default lm_evaluation_harness provider
        """
        result = call_mcp_tool(
            client=evalhub_mcp_client,
            name="discover_providers",
            arguments={},
        )
        structured = mcp_tool_structured(result=result)
        provider_ids = {item.get("id") for item in structured.get("providers", []) if isinstance(item, dict)}
        assert EVALHUB_MCP_DEFAULT_PROVIDER_ID in provider_ids

    @pytest.mark.parametrize(
        "arguments,expected_fragment",
        [
            pytest.param({}, "benchmarks", id="test_missing_benchmarks_and_collection"),
            pytest.param(
                {
                    "name": "invalid-job",
                    "model": {"url": "http://model/v1", "name": "m"},
                    "benchmarks": [{"id": "arc_easy", "provider_id": "lm_evaluation_harness"}],
                    "collection": {"id": EVALHUB_MCP_DEFAULT_COLLECTION_ID},
                },
                "not both",
                id="test_benchmarks_and_collection",
            ),
        ],
    )
    def test_submit_evaluation_validation_errors(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
        arguments: dict,
        expected_fragment: str,
    ) -> None:
        """
        Given: Authenticated MCP client with invalid submit_evaluation arguments
        When: submit_evaluation tool is invoked
        Then: Tool returns a validation error containing the expected fragment
        """
        result = call_mcp_tool(
            client=evalhub_mcp_client,
            name="submit_evaluation",
            arguments=arguments,
        )
        assert mcp_tool_is_error(result=result), f"Expected tool error result, got: {result}"
        text = mcp_tool_error_text(result=result)
        assert expected_fragment.lower() in text.lower(), f"Expected '{expected_fragment}' in error: {text}"


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-mcp-tools-submit"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier2
@pytest.mark.ai_safety
class TestEvalHubMcpToolsSubmit:
    """MCP submit_evaluation success paths."""

    def test_submit_evaluation_with_benchmarks(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
        evalhub_vllm_emulator_service: Service,
        tenant_a_namespace: Namespace,
    ) -> None:
        """
        Given: Authenticated MCP client with a vLLM emulator model URL
        When: submit_evaluation is called with benchmarks
        Then: Response returns a pending or running job ID
        """
        model_url = build_mcp_model_url(
            service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
        )
        result = call_mcp_tool(
            client=evalhub_mcp_client,
            name="submit_evaluation",
            arguments=build_mcp_evaluation_arguments(
                model_url=model_url,
                job_name=f"mcp-benchmark-submit-{uuid.uuid4().hex[:8]}",
            ),
        )
        assert not mcp_tool_is_error(result=result), mcp_tool_error_text(result=result)
        structured = mcp_tool_structured(result=result)
        assert structured.get("job_id")
        assert structured.get("state") in {"pending", "running", "completed"}

    def test_submit_evaluation_with_collection(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
        evalhub_vllm_emulator_service: Service,
        tenant_a_namespace: Namespace,
    ) -> None:
        """
        Given: Authenticated MCP client with a vLLM emulator model URL
        When: submit_evaluation is called with a collection ID
        Then: Response returns a pending or running job ID
        """
        model_url = build_mcp_model_url(
            service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
        )
        result = call_mcp_tool(
            client=evalhub_mcp_client,
            name="submit_evaluation",
            arguments=build_mcp_evaluation_arguments(
                model_url=model_url,
                job_name=f"mcp-collection-submit-{uuid.uuid4().hex[:8]}",
                collection_id=EVALHUB_MCP_DEFAULT_COLLECTION_ID,
            ),
        )
        assert not mcp_tool_is_error(result=result), mcp_tool_error_text(result=result)
        structured = mcp_tool_structured(result=result)
        assert structured.get("job_id")
