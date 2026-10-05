import pytest
import requests
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route

from tests.ai_safety.evalhub.constants import (
    EVALHUB_PROVIDERS_PATH,
    LM_EVALUATION_HARNESS_PROVIDER_ID,
)
from tests.ai_safety.evalhub.utils import build_headers


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-providers-agentic"},
        ),
    ],
    indirect=True,
)
@pytest.mark.ai_safety
@pytest.mark.usefixtures("evalhub_cr")
class TestEvalHubProvidersAgentic:
    """Tests for the agent metadata fields on EvalHub providers and benchmarks.

    Validates that system providers expose the agent block (target_type, evaluates,
    summary, score_ranges, etc.) in both the list and single-resource endpoints.
    """

    _VALID_TARGET_TYPES: frozenset[str] = frozenset({"model", "agent", "inference_server"})

    @pytest.mark.tier1
    def test_lmeval_provider_has_agent_metadata(
        self,
        current_client_token: str,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
        model_namespace: Namespace,
    ) -> None:
        """lm_evaluation_harness provider exposes a non-null agent metadata block."""
        url = f"https://{evalhub_route.host}{EVALHUB_PROVIDERS_PATH}?limit=100"
        response = requests.get(
            url=url,
            headers=build_headers(token=current_client_token, tenant=model_namespace.name),
            verify=evalhub_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        items = response.json().get("items", [])
        provider = next(
            (p for p in items if p.get("resource", {}).get("id") == LM_EVALUATION_HARNESS_PROVIDER_ID),
            None,
        )
        assert provider is not None, f"Provider '{LM_EVALUATION_HARNESS_PROVIDER_ID}' not found in providers list"
        assert provider.get("agent") is not None, (
            f"Provider '{LM_EVALUATION_HARNESS_PROVIDER_ID}' missing 'agent' metadata block"
        )

    @pytest.mark.tier2
    def test_lmeval_agent_target_type_is_model(
        self,
        current_client_token: str,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
        model_namespace: Namespace,
    ) -> None:
        """lm_evaluation_harness agent.target_type is 'model' (evaluates LLMs, not agents)."""
        url = f"https://{evalhub_route.host}{EVALHUB_PROVIDERS_PATH}/{LM_EVALUATION_HARNESS_PROVIDER_ID}"
        response = requests.get(
            url=url,
            headers=build_headers(token=current_client_token, tenant=model_namespace.name),
            verify=evalhub_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        agent = response.json().get("agent", {})
        assert agent, f"Provider '{LM_EVALUATION_HARNESS_PROVIDER_ID}' has no 'agent' field"
        assert agent.get("target_type") == "model", f"Expected target_type='model', got '{agent.get('target_type')}'"

    @pytest.mark.tier2
    def test_lmeval_agent_evaluates_is_non_empty(
        self,
        current_client_token: str,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
        model_namespace: Namespace,
    ) -> None:
        """lm_evaluation_harness agent.evaluates is a non-empty list containing standard categories."""
        url = f"https://{evalhub_route.host}{EVALHUB_PROVIDERS_PATH}/{LM_EVALUATION_HARNESS_PROVIDER_ID}"
        response = requests.get(
            url=url,
            headers=build_headers(token=current_client_token, tenant=model_namespace.name),
            verify=evalhub_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        evaluates = response.json().get("agent", {}).get("evaluates", [])
        assert isinstance(evaluates, list) and evaluates, f"Expected non-empty 'evaluates' list, got: {evaluates}"
        for expected_category in ("accuracy", "reasoning"):
            assert expected_category in evaluates, (
                f"Expected category '{expected_category}' in agent.evaluates, got: {evaluates}"
            )

    @pytest.mark.tier2
    def test_lmeval_agent_summary_present_and_bounded(
        self,
        current_client_token: str,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
        model_namespace: Namespace,
    ) -> None:
        """lm_evaluation_harness agent.summary is a non-empty string of at most 200 characters."""
        url = f"https://{evalhub_route.host}{EVALHUB_PROVIDERS_PATH}/{LM_EVALUATION_HARNESS_PROVIDER_ID}"
        response = requests.get(
            url=url,
            headers=build_headers(token=current_client_token, tenant=model_namespace.name),
            verify=evalhub_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        summary = response.json().get("agent", {}).get("summary", "")
        assert isinstance(summary, str) and summary, f"Expected non-empty string for agent.summary, got: {summary!r}"
        assert len(summary) <= 200, f"agent.summary exceeds 200 characters ({len(summary)}): {summary!r}"

    @pytest.mark.tier2
    def test_all_providers_with_agent_have_valid_target_type(
        self,
        current_client_token: str,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
        model_namespace: Namespace,
    ) -> None:
        """Every provider that includes an agent block has a valid target_type value."""
        url = f"https://{evalhub_route.host}{EVALHUB_PROVIDERS_PATH}?limit=100"
        response = requests.get(
            url=url,
            headers=build_headers(token=current_client_token, tenant=model_namespace.name),
            verify=evalhub_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        invalid = []
        for item in response.json().get("items", []):
            agent = item.get("agent")
            if agent is None:
                continue
            provider_id = item.get("resource", {}).get("id", "<unknown>")
            target_type = agent.get("target_type")
            if target_type not in self._VALID_TARGET_TYPES:
                invalid.append(f"{provider_id}: target_type={target_type!r}")

        assert not invalid, f"Providers with invalid target_type (must be one of {self._VALID_TARGET_TYPES}): {invalid}"

    @pytest.mark.tier2
    def test_single_provider_endpoint_exposes_agent_metadata(
        self,
        current_client_token: str,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
        model_namespace: Namespace,
    ) -> None:
        """GET /providers/{id} for lm_evaluation_harness returns 200 with agent metadata."""
        url = f"https://{evalhub_route.host}{EVALHUB_PROVIDERS_PATH}/{LM_EVALUATION_HARNESS_PROVIDER_ID}"
        response = requests.get(
            url=url,
            headers=build_headers(token=current_client_token, tenant=model_namespace.name),
            verify=evalhub_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 200, (
            f"Expected 200 for GET /providers/{LM_EVALUATION_HARNESS_PROVIDER_ID}, "
            f"got {response.status_code}: {response.text}"
        )
        body = response.json()
        assert body.get("resource", {}).get("id") == LM_EVALUATION_HARNESS_PROVIDER_ID
        assert body.get("agent") is not None, "Single-provider endpoint did not return 'agent' metadata block"
        assert body["agent"].get("target_type") == "model", (
            f"Expected agent.target_type='model', got '{body['agent'].get('target_type')}'"
        )

    @pytest.mark.tier2
    def test_benchmark_agent_score_ranges(
        self,
        current_client_token: str,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
        model_namespace: Namespace,
    ) -> None:
        """arc_easy benchmark within lm_evaluation_harness exposes agent.score_ranges."""
        url = f"https://{evalhub_route.host}{EVALHUB_PROVIDERS_PATH}/{LM_EVALUATION_HARNESS_PROVIDER_ID}"
        response = requests.get(
            url=url,
            headers=build_headers(token=current_client_token, tenant=model_namespace.name),
            verify=evalhub_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        benchmarks = response.json().get("benchmarks", [])
        arc_easy = next((b for b in benchmarks if b.get("id") == "arc_easy"), None)
        assert arc_easy is not None, "Benchmark 'arc_easy' not found in lm_evaluation_harness provider"

        score_ranges = arc_easy.get("agent", {}).get("score_ranges", [])
        assert isinstance(score_ranges, list) and score_ranges, (
            f"Expected non-empty score_ranges for arc_easy benchmark, got: {score_ranges}"
        )
        for entry in score_ranges:
            assert "range" in entry, f"score_range entry missing 'range' field: {entry}"
            assert "meaning" in entry, f"score_range entry missing 'meaning' field: {entry}"
            assert isinstance(entry["range"], str), f"'range' must be a string, got: {entry['range']!r}"
            assert isinstance(entry["meaning"], str), f"'meaning' must be a string, got: {entry['meaning']!r}"
