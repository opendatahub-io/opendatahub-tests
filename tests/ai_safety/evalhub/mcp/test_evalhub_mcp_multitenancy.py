import uuid

import pytest
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.service import Service

from tests.ai_safety.evalhub.mcp.utils import (
    EvalHubMcpClient,
    build_mcp_evaluation_arguments,
    build_mcp_model_url,
    call_mcp_tool,
    mcp_tool_is_error,
    submit_evaluation_via_mcp,
)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-mcp-multitenancy"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier2
@pytest.mark.ai_safety
@pytest.mark.usefixtures("evalhub_mcp_proxy_role_binding")
class TestEvalHubMcpMultitenancy:
    """MCP multi-tenancy behavior with tenant-scoped EvalHub configuration.

    The MCP deployment is configured with EVALHUB_TENANT=tenant-a (outbound API scope).
    Inbound X-Tenant is required by kube-rbac-proxy but outbound jobs are created
    in the configured tenant namespace regardless of the inbound header value.
    """

    def test_mcp_submit_with_configured_tenant_succeeds(
        self,
        evalhub_mcp_client: EvalHubMcpClient,
        tenant_a_namespace: Namespace,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """
        Given: MCP client authenticated for tenant-a with proxy RBAC
        When: submit_evaluation is called with a tenant-a model URL
        Then: Job is created and job_id is returned
        """
        model_url = build_mcp_model_url(
            service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
        )
        submit_result = submit_evaluation_via_mcp(
            client=evalhub_mcp_client,
            arguments=build_mcp_evaluation_arguments(
                model_url=model_url,
                job_name=f"mcp-mt-submit-{uuid.uuid4().hex[:8]}",
            ),
        )
        assert submit_result.get("job_id")

    def test_mcp_submit_with_mismatched_tenant_header_still_creates_job(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        tenant_b_namespace: Namespace,
        evalhub_mcp_mt_route: Route,
        evalhub_mcp_mt_ca_bundle_file: str,
        evalhub_vllm_emulator_service: Service,
        evalhub_mcp_mt_ready: None,
    ) -> None:
        """
        Given: MCP deployment scoped to tenant-a with mismatched inbound X-Tenant
        When: submit_evaluation is called via MCP tools
        Then: Job submission succeeds using outbound tenant-a scope
        """
        client = EvalHubMcpClient(
            host=evalhub_mcp_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mcp_mt_ca_bundle_file,
            tenant=tenant_b_namespace.name,
        )
        client.initialize()

        model_url = build_mcp_model_url(
            service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
        )
        result = call_mcp_tool(
            client=client,
            name="submit_evaluation",
            arguments=build_mcp_evaluation_arguments(
                model_url=model_url,
                job_name=f"mcp-mt-mismatch-{uuid.uuid4().hex[:8]}",
            ),
        )
        assert not mcp_tool_is_error(result=result), (
            "Expected submit to succeed because outbound tenant is tenant-a "
            f"regardless of inbound X-Tenant={tenant_b_namespace.name}"
        )
