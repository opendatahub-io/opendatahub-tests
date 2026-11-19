"""Provider discovery tests for IBM CLEAR (RHOAIENG-63558).

Given: EvalHub with MLflow configured and CLEAR provider available
When:  Providers API is queried for ibm-clear / FVT CLEAR provider
Then:  Provider is listed and exposes the agentic-evaluation benchmark
"""

from __future__ import annotations

import pytest
import requests
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route

from tests.ai_safety.evalhub.clear.constants import CLEAR_BENCHMARK_ID, IBM_CLEAR_PROVIDER_ID
from tests.ai_safety.evalhub.clear.utils import get_provider
from tests.ai_safety.evalhub.constants import EVALHUB_PROVIDERS_PATH
from tests.ai_safety.evalhub.utils import build_headers, validate_evalhub_health


@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-evalhub-clear-provider"})],
    indirect=True,
)
@pytest.mark.ai_safety
@pytest.mark.usefixtures("evalhub_mt_ready")
class TestEvalHubClearProvider:
    """C-01: CLEAR provider discovery and agentic benchmark metadata."""

    @pytest.mark.tier1
    def test_evalhub_health_with_mlflow(
        self,
        tenant_a_token: str,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """Given EvalHub with MLFLOW_TRACKING_URI set, when health is checked, then it is healthy."""
        validate_evalhub_health(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
        )

    @pytest.mark.tier1
    def test_clear_provider_available(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        clear_provider_id: str,
    ) -> None:
        """Given CLEAR is registered, when providers are listed, then the CLEAR provider id is present."""
        url = f"https://{evalhub_mt_route.host}{EVALHUB_PROVIDERS_PATH}?limit=100"
        response = requests.get(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        provider_ids = [item.get("resource", {}).get("id") for item in response.json().get("items", [])]
        assert clear_provider_id in provider_ids, (
            f"Expected CLEAR provider '{clear_provider_id}' in providers list, got: {provider_ids}"
        )

    @pytest.mark.tier1
    def test_clear_provider_exposes_agentic_evaluation_benchmark(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        clear_provider_id: str,
    ) -> None:
        """Given CLEAR provider, when GET /providers/{id}, then agentic-evaluation benchmark is listed."""
        body = get_provider(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            provider_id=clear_provider_id,
        )
        benchmarks = body.get("benchmarks", [])
        benchmark_ids = [benchmark.get("id") for benchmark in benchmarks]
        assert CLEAR_BENCHMARK_ID in benchmark_ids, (
            f"Expected benchmark '{CLEAR_BENCHMARK_ID}' on CLEAR provider, got: {benchmark_ids}"
        )

    @pytest.mark.tier2
    def test_system_ibm_clear_id_when_bundled(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        clear_provider_id: str,
    ) -> None:
        """When the operator ships system ibm-clear, the resolved provider id matches the catalog id."""
        if clear_provider_id != IBM_CLEAR_PROVIDER_ID:
            pytest.skip("Cluster does not expose system provider 'ibm-clear'; using FVT user provider instead")
        body = get_provider(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            provider_id=IBM_CLEAR_PROVIDER_ID,
        )
        assert body.get("resource", {}).get("id") == IBM_CLEAR_PROVIDER_ID
