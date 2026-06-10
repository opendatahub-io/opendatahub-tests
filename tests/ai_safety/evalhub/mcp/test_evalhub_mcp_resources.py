import json
import uuid

import pytest
from ocp_resources.namespace import Namespace
from ocp_resources.service import Service

from tests.ai_safety.evalhub.mcp.constants import (
    EVALHUB_MCP_BENCHMARK_URI_TEMPLATE,
    EVALHUB_MCP_COLLECTION_URI_TEMPLATE,
    EVALHUB_MCP_DEFAULT_BENCHMARK_ID,
    EVALHUB_MCP_DEFAULT_COLLECTION_ID,
    EVALHUB_MCP_DEFAULT_PROVIDER_ID,
    EVALHUB_MCP_JOB_URI_TEMPLATE,
    EVALHUB_MCP_JOBS_BY_STATUS_URI_TEMPLATE,
    EVALHUB_MCP_PROVIDER_URI_TEMPLATE,
)
from tests.ai_safety.evalhub.mcp.utils import (
    EvalHubMcpClient,
    McpProtocolError,
    build_mcp_evaluation_arguments,
    build_mcp_model_url,
    mcp_read_resource_text,
    read_mcp_resource,
    submit_evaluation_via_mcp,
)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-mcp-resources"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.ai_safety
class TestEvalHubMcpResources:
    """MCP dynamic resource template tests (evalhub://providers/{id}, etc.)."""

    def test_read_provider_by_id(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """resources/read on evalhub://providers/{id} returns provider metadata."""
        uri = EVALHUB_MCP_PROVIDER_URI_TEMPLATE.format(provider_id=EVALHUB_MCP_DEFAULT_PROVIDER_ID)
        result = read_mcp_resource(client=evalhub_mcp_client, uri=uri)
        payload = json.loads(mcp_read_resource_text(result=result))
        assert payload.get("resource", {}).get("id") == EVALHUB_MCP_DEFAULT_PROVIDER_ID

    def test_read_benchmark_by_id(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """resources/read on evalhub://benchmarks/{id} returns benchmark metadata."""
        uri = EVALHUB_MCP_BENCHMARK_URI_TEMPLATE.format(benchmark_id=EVALHUB_MCP_DEFAULT_BENCHMARK_ID)
        result = read_mcp_resource(client=evalhub_mcp_client, uri=uri)
        payload = json.loads(mcp_read_resource_text(result=result))
        assert payload.get("resource", {}).get("id") == EVALHUB_MCP_DEFAULT_BENCHMARK_ID

    def test_read_collection_by_id(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """resources/read on evalhub://collections/{id} returns the configured collection."""
        uri = EVALHUB_MCP_COLLECTION_URI_TEMPLATE.format(collection_id=EVALHUB_MCP_DEFAULT_COLLECTION_ID)
        result = read_mcp_resource(client=evalhub_mcp_client, uri=uri)
        payload = json.loads(mcp_read_resource_text(result=result))
        assert payload.get("resource", {}).get("id") == EVALHUB_MCP_DEFAULT_COLLECTION_ID

    def test_read_nonexistent_provider_returns_error(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """resources/read for an unknown provider ID returns an MCP protocol error."""
        uri = EVALHUB_MCP_PROVIDER_URI_TEMPLATE.format(provider_id="nonexistent-provider-id")
        with pytest.raises(McpProtocolError):
            read_mcp_resource(client=evalhub_mcp_client, uri=uri)

    def test_read_nonexistent_benchmark_returns_error(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
    ) -> None:
        """resources/read for an unknown benchmark ID returns an MCP protocol error."""
        uri = EVALHUB_MCP_BENCHMARK_URI_TEMPLATE.format(benchmark_id="nonexistent-benchmark-id")
        with pytest.raises(McpProtocolError):
            read_mcp_resource(client=evalhub_mcp_client, uri=uri)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-mcp-resources-jobs"},
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
class TestEvalHubMcpJobResources:
    """MCP job resource reads after submitting an evaluation via tools."""

    def test_read_job_by_id_after_submit(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
        evalhub_mcp_vllm_emulator_service: Service,
        tenant_a_namespace: Namespace,
    ) -> None:
        """resources/read on evalhub://jobs/{id} returns the submitted job."""
        model_url = build_mcp_model_url(
            service_name=evalhub_mcp_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
        )
        submit_result = submit_evaluation_via_mcp(
            client=evalhub_mcp_client,
            arguments=build_mcp_evaluation_arguments(
                model_url=model_url,
                job_name=f"mcp-resource-job-{uuid.uuid4().hex[:8]}",
            ),
        )
        job_id = submit_result["job_id"]

        uri = EVALHUB_MCP_JOB_URI_TEMPLATE.format(job_id=job_id)
        result = read_mcp_resource(client=evalhub_mcp_client, uri=uri)
        payload = json.loads(mcp_read_resource_text(result=result))
        assert payload.get("resource", {}).get("id") == job_id

    def test_jobs_list_includes_submitted_job(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
        evalhub_mcp_vllm_emulator_service: Service,
        tenant_a_namespace: Namespace,
    ) -> None:
        """resources/read on evalhub://jobs lists jobs created via MCP tools."""
        model_url = build_mcp_model_url(
            service_name=evalhub_mcp_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
        )
        submit_result = submit_evaluation_via_mcp(
            client=evalhub_mcp_client,
            arguments=build_mcp_evaluation_arguments(
                model_url=model_url,
                job_name=f"mcp-jobs-list-{uuid.uuid4().hex[:8]}",
            ),
        )
        job_id = submit_result["job_id"]

        result = read_mcp_resource(client=evalhub_mcp_client, uri="evalhub://jobs")
        jobs = json.loads(mcp_read_resource_text(result=result))
        job_ids = [item.get("resource", {}).get("id") for item in jobs if isinstance(item, dict)]
        assert job_id in job_ids, f"Submitted job {job_id} not found in jobs resource: {job_ids}"

    def test_jobs_filtered_by_pending_status(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
        evalhub_mcp_vllm_emulator_service: Service,
        tenant_a_namespace: Namespace,
    ) -> None:
        """resources/read on evalhub://jobs?status=pending returns only pending jobs."""
        model_url = build_mcp_model_url(
            service_name=evalhub_mcp_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
        )
        submit_result = submit_evaluation_via_mcp(
            client=evalhub_mcp_client,
            arguments=build_mcp_evaluation_arguments(
                model_url=model_url,
                job_name=f"mcp-pending-filter-{uuid.uuid4().hex[:8]}",
            ),
        )
        job_id = submit_result["job_id"]

        uri = EVALHUB_MCP_JOBS_BY_STATUS_URI_TEMPLATE.format(status="pending")
        result = read_mcp_resource(client=evalhub_mcp_client, uri=uri)
        jobs = json.loads(mcp_read_resource_text(result=result))
        job_ids = [item.get("resource", {}).get("id") for item in jobs if isinstance(item, dict)]
        assert job_id in job_ids, f"Pending job {job_id} not found in filtered list: {job_ids}"
