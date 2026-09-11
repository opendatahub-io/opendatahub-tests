"""End-to-end model recommendation workflow via rhoai-mcp MCP tools.

Exercises the Planner composite tools (recommend_model, get_deployment_config)
running in LOCAL mode — the default, which embeds the llm-d-planner library
with an in-memory SQLite database.  When the Model Catalog service is
available on the cluster, benchmark data is synced from it; otherwise the
planner falls back to bundled BLIS benchmarks (a warning is emitted at
fixture setup time).
"""

import pytest
import yaml
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError

from tests.rhoai_mcp.constants import (
    RHOAI_MCP_NAMESPACE,
    RHOAI_MCP_RECOMMEND_GPU_TYPES,
    RHOAI_MCP_RECOMMEND_USE_CASE,
    RHOAI_MCP_RECOMMEND_USER_COUNT,
)
from tests.rhoai_mcp.utils import parse_tool_result

_RECOMMENDATION_CATEGORIES = ("top_performance", "top_cost", "top_balanced", "top_quality")


@pytest.mark.asyncio
@pytest.mark.tier1
class TestRhoaiMcpModelRecommendation:
    """Validate model recommendation workflow using rhoai-mcp MCP tools.

    Steps:
        1. Call recommend_model with chatbot use case and explicit overrides.
        2. Verify specification structure (use_case, SLO targets, traffic profile).
        3. Verify recommendation categories are populated with valid structure.
        4. Verify SLO overrides are honoured.
        5. Test validation error handling for invalid inputs.
        6. Call get_deployment_config with specification values from step 1.
        7. Verify generated configs contain valid Kubernetes YAML.
        8. Validate generated YAML via K8s API server-side dry-run.
        9. Test validation error handling for get_deployment_config.
    """

    _recommend_data: dict | None = None
    _deploy_config_data: dict | None = None

    @pytest.mark.dependency(name="recommend_model")
    async def test_recommend_model(
        self,
        rhoai_mcp_transport: StreamableHttpTransport,
    ) -> None:
        """Given the planner is running in local mode
        When recommend_model is called with explicit overrides
        Then the response contains a specification and recommendations

        ``text`` is required by the MCP tool API but ignored by the local
        planner backend — intent extraction from natural language is not
        supported in local mode.  All parameters must be provided as
        explicit overrides.
        """
        async with Client(rhoai_mcp_transport) as client:
            result = await client.call_tool(
                name="recommend_model",
                arguments={
                    "text": "overrides provided",
                    "use_case": RHOAI_MCP_RECOMMEND_USE_CASE,
                    "user_count": RHOAI_MCP_RECOMMEND_USER_COUNT,
                    "preferred_gpu_types": RHOAI_MCP_RECOMMEND_GPU_TYPES,
                },
            )
        data = parse_tool_result(result=result)

        assert "error" not in data, f"recommend_model returned error: {data}"
        assert "specification" in data, f"Missing 'specification' key: {data.keys()}"
        assert "recommendations" in data, f"Missing 'recommendations' key: {data.keys()}"

        TestRhoaiMcpModelRecommendation._recommend_data = data

    @pytest.mark.dependency(depends=["recommend_model"])
    async def test_recommend_specification_structure(self) -> None:
        """Given a successful recommend_model response
        When the specification is inspected
        Then it contains the echoed overrides and valid SLO/traffic ranges
        """
        spec = self._recommend_data["specification"]

        assert spec["use_case"] == RHOAI_MCP_RECOMMEND_USE_CASE
        assert spec["user_count"] == RHOAI_MCP_RECOMMEND_USER_COUNT

        slo = spec["slo_targets"]
        assert 1 < slo["ttft_target_ms"] < 10_000, f"ttft_target_ms out of range: {slo['ttft_target_ms']}"
        assert 1 < slo["itl_target_ms"] < 1_000, f"itl_target_ms out of range: {slo['itl_target_ms']}"
        assert 1 < slo["e2e_target_ms"] < 100_000, f"e2e_target_ms out of range: {slo['e2e_target_ms']}"

        tp = spec["traffic_profile"]
        assert tp["prompt_tokens"] > 0
        assert tp["output_tokens"] > 0
        assert tp["expected_qps"] > 0

    @pytest.mark.dependency(name="recommend_has_categories", depends=["recommend_model"])
    async def test_recommend_has_categories(self) -> None:
        """Given a successful recommend_model response
        When recommendations are inspected
        Then at least one category is populated
        """
        recs = self._recommend_data["recommendations"]
        if not recs:
            pytest.skip("No recommendations with bundled benchmark data")

        has_any = any(cat in recs for cat in _RECOMMENDATION_CATEGORIES)
        assert has_any, f"No recommendation categories populated: {recs.keys()}"

    @pytest.mark.dependency(depends=["recommend_has_categories"])
    async def test_recommend_entry_structure(self) -> None:
        """Given populated recommendation categories
        When each entry is inspected
        Then it has model, meets_slo, gpu, and optional cost/score fields in valid ranges
        """
        recs = self._recommend_data["recommendations"]

        for key, entry in recs.items():
            assert isinstance(entry.get("model"), str) and entry["model"], f"'{key}' missing or empty 'model'"
            assert isinstance(entry.get("meets_slo"), bool), f"'{key}' missing or non-boolean 'meets_slo'"
            assert isinstance(entry.get("gpu"), str) and entry["gpu"], f"'{key}' missing or empty 'gpu'"
            if "cost_usd_month" in entry:
                assert entry["cost_usd_month"] > 0, f"'{key}' cost_usd_month not positive"
            if "score" in entry:
                assert 0 <= entry["score"] <= 100, f"'{key}' score out of [0, 100] range"

    async def test_recommend_slo_overrides(
        self,
        rhoai_mcp_transport: StreamableHttpTransport,
    ) -> None:
        """Given explicit SLO override values
        When recommend_model is called with those overrides
        Then the specification echoes the exact override values
        """
        async with Client(rhoai_mcp_transport) as client:
            result = await client.call_tool(
                name="recommend_model",
                arguments={
                    "text": "overrides provided",
                    "use_case": RHOAI_MCP_RECOMMEND_USE_CASE,
                    "user_count": RHOAI_MCP_RECOMMEND_USER_COUNT,
                    "preferred_gpu_types": RHOAI_MCP_RECOMMEND_GPU_TYPES,
                    "ttft_max_ms": 100,
                    "itl_max_ms": 30,
                    "e2e_max_ms": 1500,
                },
            )
        data = parse_tool_result(result=result)
        assert "error" not in data, f"recommend_model returned error: {data}"

        slo = data["specification"]["slo_targets"]
        assert slo["ttft_target_ms"] == 100
        assert slo["itl_target_ms"] == 30
        assert slo["e2e_target_ms"] == 1500

    async def test_recommend_validation_errors(
        self,
        rhoai_mcp_transport: StreamableHttpTransport,
    ) -> None:
        """Given invalid input values
        When recommend_model is called
        Then each call returns an error dict with a relevant message
        """
        cases = [
            ({"text": "test", "use_case": "invalid_case"}, "use_case"),
            ({"text": "test", "min_quality": 101}, "min_quality"),
            ({"text": "test", "preferred_gpu_types": ["V100"]}, "V100"),
            ({"text": ""}, "text"),
        ]
        async with Client(rhoai_mcp_transport) as client:
            for arguments, expected_substr in cases:
                result = await client.call_tool(
                    name="recommend_model",
                    arguments=arguments,
                )
                data = parse_tool_result(result=result)
                assert "error" in data, f"Expected error for {arguments}, got: {data}"
                assert expected_substr in data["error"], (
                    f"Expected '{expected_substr}' in error message, got: {data['error']}"
                )

    @pytest.mark.dependency(name="get_deployment_config", depends=["recommend_model"])
    async def test_get_deployment_config(
        self,
        rhoai_mcp_transport: StreamableHttpTransport,
    ) -> None:
        """Given a successful recommend_model specification
        When get_deployment_config is called with those values for balanced category
        Then the response contains deployment_id, namespace, and configs
        """
        spec = self._recommend_data["specification"]
        slo = spec["slo_targets"]
        tp = spec["traffic_profile"]

        async with Client(rhoai_mcp_transport) as client:
            result = await client.call_tool(
                name="get_deployment_config",
                arguments={
                    "category": "balanced",
                    "use_case": spec["use_case"],
                    "user_count": spec["user_count"],
                    "prompt_tokens": tp["prompt_tokens"],
                    "output_tokens": tp["output_tokens"],
                    "expected_qps": tp["expected_qps"],
                    "ttft_target_ms": slo["ttft_target_ms"],
                    "itl_target_ms": slo["itl_target_ms"],
                    "e2e_target_ms": slo["e2e_target_ms"],
                    "namespace": RHOAI_MCP_NAMESPACE,
                },
            )
        data = parse_tool_result(result=result)

        if "error" in data and data.get("status_code") == 404:
            pytest.skip("No recommendation found for balanced category with bundled benchmarks")

        assert "error" not in data, f"get_deployment_config returned error: {data}"
        assert data.get("deployment_id"), "Missing or empty deployment_id"
        assert data["namespace"] == RHOAI_MCP_NAMESPACE
        assert data.get("configs"), "Missing or empty configs dict"

        TestRhoaiMcpModelRecommendation._deploy_config_data = data

    @pytest.mark.dependency(depends=["get_deployment_config"])
    async def test_deploy_config_has_yaml(self) -> None:
        """Given a successful get_deployment_config response
        When the configs are inspected
        Then at least one contains an InferenceService manifest
        """
        configs = self._deploy_config_data["configs"]

        assert any("inferenceservice" in k.lower() for k in configs), (
            f"No InferenceService config found in keys: {list(configs.keys())}"
        )
        for name, content in configs.items():
            assert isinstance(content, str), f"Config '{name}' is not a string"
            assert "apiVersion" in content or "kind" in content, (
                f"Config '{name}' does not look like a Kubernetes manifest"
            )

    @pytest.mark.dependency(depends=["get_deployment_config"])
    async def test_deploy_config_server_dry_run(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Given generated Kubernetes YAML configs
        When each manifest is submitted as a server-side dry-run
        Then the K8s API server accepts them without validation errors
        """
        configs = self._deploy_config_data["configs"]
        validated = 0

        for content in configs.values():
            for manifest in yaml.safe_load_all(content):
                if not isinstance(manifest, dict):
                    continue
                api_version = manifest.get("apiVersion")
                kind = manifest.get("kind")
                if not api_version or not kind:
                    continue

                try:
                    resource = admin_client.resources.get(api_version=api_version, kind=kind)
                except ResourceNotFoundError:
                    continue

                namespace = manifest.get("metadata", {}).get("namespace", RHOAI_MCP_NAMESPACE)
                resource.create(body=manifest, namespace=namespace, dry_run="All")
                validated += 1

        assert validated > 0, f"No configs could be validated via dry-run; keys: {list(configs.keys())}"

    async def test_deploy_config_validation_errors(
        self,
        rhoai_mcp_transport: StreamableHttpTransport,
    ) -> None:
        """Given invalid input values
        When get_deployment_config is called
        Then each call returns an error dict with a relevant message
        """
        base = {
            "use_case": RHOAI_MCP_RECOMMEND_USE_CASE,
            "user_count": RHOAI_MCP_RECOMMEND_USER_COUNT,
            "prompt_tokens": 512,
            "output_tokens": 256,
            "expected_qps": 10.0,
            "ttft_target_ms": 200,
            "itl_target_ms": 65,
            "e2e_target_ms": 5000,
        }
        cases = [
            ({**base, "category": "fastest"}, "category"),
            ({**base, "category": "balanced", "namespace": "INVALID!"}, "namespace"),
            ({**base, "category": "balanced", "user_count": 0}, "user_count"),
        ]
        async with Client(rhoai_mcp_transport) as client:
            for arguments, expected_substr in cases:
                result = await client.call_tool(
                    name="get_deployment_config",
                    arguments=arguments,
                )
                data = parse_tool_result(result=result)
                assert "error" in data, f"Expected error for {arguments}, got: {data}"
                assert expected_substr in data["error"], (
                    f"Expected '{expected_substr}' in error message, got: {data['error']}"
                )
