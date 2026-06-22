import pytest
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.service import Service

from tests.ai_safety.evalhub.mcp.utils import build_mcp_evaluation_arguments
from tests.ai_safety.evalhub.utils import (
    submit_evalhub_job,
    validate_evalhub_job_completed,
    wait_for_evalhub_job,
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
@pytest.mark.ai_safety
class TestEvalHubMcpE2E:
    """End-to-end tests for EvalHub MCP evaluation submission.

    Submits an lm_evaluation_harness job (arc_easy) via the MCP
    ``submit_evaluation`` arguments builder against a vLLM emulator
    endpoint.  Validates that benchmark parameters (including tokenizer)
    are propagated so the job completes successfully.
    """

    def test_mcp_evaluation_submit_and_complete(
        self,
        mcp_tenant_token: str,
        mcp_tenant_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        mcp_vllm_emulator_service: Service,
        evalhub_mcp_ready: None,
    ) -> None:
        """Given an OpenAI-compatible model endpoint with a vLLM emulator,
        when an evaluation is submitted via the MCP arguments builder
        including benchmark parameters with a tokenizer,
        then the job reaches 'completed' state with benchmark results.
        """
        payload = build_mcp_evaluation_arguments(
            model_service_name=mcp_vllm_emulator_service.name,
            tenant_namespace=mcp_tenant_namespace.name,
        )
        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=mcp_tenant_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=mcp_tenant_namespace.name,
            payload=payload,
        )
        job_id = data["resource"]["id"]

        job_result = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=mcp_tenant_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=mcp_tenant_namespace.name,
            job_id=job_id,
            timeout=600,
        )
        terminal_state = job_result.get("status", {}).get("state")
        assert terminal_state == "completed", (
            f"Expected job state 'completed', got '{terminal_state}'. "
            "Verify that benchmark.parameters includes tokenizer for OpenAI-compatible endpoints."
        )
        validate_evalhub_job_completed(job_data=job_result)

    def test_mcp_evaluation_payload_includes_parameters(
        self,
        mcp_tenant_namespace: Namespace,
        mcp_vllm_emulator_service: Service,
    ) -> None:
        """Given the MCP evaluation arguments builder,
        when building a payload for a vLLM emulator endpoint,
        then the benchmarks contain a parameters dict with tokenizer.
        """
        payload = build_mcp_evaluation_arguments(
            model_service_name=mcp_vllm_emulator_service.name,
            tenant_namespace=mcp_tenant_namespace.name,
        )
        benchmarks = payload.get("benchmarks", [])
        assert len(benchmarks) == 1, f"Expected 1 benchmark, got {len(benchmarks)}"

        parameters = benchmarks[0].get("parameters")
        assert parameters is not None, (
            "benchmark.parameters is missing from MCP evaluation arguments; "
            "OpenAI-compatible endpoints require a tokenizer parameter"
        )
        assert "tokenizer" in parameters, (
            f"benchmark.parameters must include 'tokenizer' for vLLM endpoints, got: {parameters}"
        )
        assert parameters["tokenizer"], "tokenizer must not be empty"
