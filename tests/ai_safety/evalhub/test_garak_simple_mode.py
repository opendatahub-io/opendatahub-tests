import pytest
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from timeout_sampler import TimeoutExpiredError

from tests.ai_safety.evalhub.constants import (
    GARAK_BENCHMARK_ID,
    GARAK_QUICK_BENCHMARK_ID,
    GARAK_SIMPLE_PROVIDER_ID,
)
from tests.ai_safety.evalhub.utils import (
    validate_evalhub_health,
    validate_evalhub_providers,
    wait_for_job_completion,
)
from utilities.general import collect_pod_information


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-garak-simple"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.usefixtures("patched_dsc_garak", "garak_evalhub_cr")
class TestGarakSimpleMode:
    """Tests for running garak security evaluation via EvalHub with simple (non-KFP) provider.

    Test order:
    1. Health check
    2. Provider availability
    3. Quick benchmark completion + results
    4. Intents benchmark completion + results
    """

    @pytest.mark.dependency(name="garak_simple_health")
    def test_evalhub_health(
        self,
        tenant_user_token: str,
        evalhub_ca_bundle_file: str,
        garak_evalhub_route: Route,
    ) -> None:
        """Verify the EvalHub service is healthy before running garak benchmark."""
        validate_evalhub_health(
            host=garak_evalhub_route.host,
            token=tenant_user_token,
            ca_bundle_file=evalhub_ca_bundle_file,
        )

    @pytest.mark.dependency(name="garak_simple_providers", depends=["garak_simple_health"])
    def test_evalhub_simple_provider_available(
        self,
        tenant_user_token: str,
        evalhub_ca_bundle_file: str,
        garak_evalhub_route: Route,
        tenant_namespace,
    ) -> None:
        """Verify the garak (simple) provider is available."""
        validate_evalhub_providers(
            host=garak_evalhub_route.host,
            token=tenant_user_token,
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant_namespace=tenant_namespace.name,
            expected_providers=[GARAK_SIMPLE_PROVIDER_ID],
        )

    @pytest.mark.dependency(name="garak_simple_quick_completes", depends=["garak_simple_providers"])
    def test_quick_garak_job_completes(
        self,
        tenant_user_token: str,
        evalhub_ca_bundle_file: str,
        garak_evalhub_route: Route,
        tenant_namespace,
        quick_garak_job_id: str,
    ) -> None:
        """Poll and verify that the quick garak job completes successfully."""
        result = wait_for_job_completion(
            host=garak_evalhub_route.host,
            token=tenant_user_token,
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant_namespace=tenant_namespace.name,
            job_id=quick_garak_job_id,
            timeout=600,
        )
        assert result, "Quick mode job completion returned empty result"
        self.__class__.quick_job_result = result

    @pytest.mark.dependency(name="garak_simple_quick_results", depends=["garak_simple_quick_completes"])
    def test_quick_garak_results(self) -> None:
        """Verify quick benchmark results have expected structure."""
        result = self.__class__.quick_job_result
        assert result, "No quick job result available"

        assert "results" in result, f"Missing 'results' in job response: {result}"

        job_results = result["results"]
        assert "benchmarks" in job_results, f"Missing 'benchmarks' in results: {job_results}"

        benchmarks = job_results["benchmarks"]
        assert len(benchmarks) > 0, "No benchmark results found"

        benchmark = benchmarks[0]
        assert benchmark.get("id") == GARAK_QUICK_BENCHMARK_ID, f"Unexpected benchmark ID: {benchmark.get('id')}"
        assert benchmark.get("provider_id") == GARAK_SIMPLE_PROVIDER_ID, (
            f"Expected garak provider, got: {benchmark.get('provider_id')}"
        )

    @pytest.mark.dependency(name="garak_simple_intents_completes", depends=["garak_simple_quick_results"])
    def test_intents_garak_job_completes(
        self,
        admin_client,
        tenant_user_token: str,
        evalhub_ca_bundle_file: str,
        garak_evalhub_route: Route,
        tenant_namespace,
        intents_garak_job_id: str,
    ) -> None:
        """Poll and verify that the intents garak evaluation job completes successfully."""
        try:
            result = wait_for_job_completion(
                host=garak_evalhub_route.host,
                token=tenant_user_token,
                ca_bundle_file=evalhub_ca_bundle_file,
                tenant_namespace=tenant_namespace.name,
                job_id=intents_garak_job_id,
                timeout=900,
            )
        except AssertionError, TimeoutExpiredError:
            for pod in Pod.get(client=admin_client, namespace=tenant_namespace.name):
                if intents_garak_job_id[:8] in pod.name:
                    collect_pod_information(pod=pod)
            raise
        assert result, "Intents mode job completion returned empty result"
        self.__class__.intents_job_result = result

    @pytest.mark.dependency(depends=["garak_simple_intents_completes"])
    def test_intents_garak_results_structure(self) -> None:
        """Verify intents job results have expected structure."""
        result = self.__class__.intents_job_result
        assert result, "No intents job result available"

        assert "results" in result, f"Missing 'results' in job response: {result}"

        job_results = result["results"]
        assert "benchmarks" in job_results, f"Missing 'benchmarks' in results: {job_results}"

        benchmarks = job_results["benchmarks"]
        assert len(benchmarks) > 0, "No benchmark results found"

        benchmark = benchmarks[0]
        assert benchmark.get("id") == GARAK_BENCHMARK_ID, f"Unexpected benchmark ID: {benchmark.get('id')}"
        assert benchmark.get("provider_id") == GARAK_SIMPLE_PROVIDER_ID, (
            f"Expected garak provider, got: {benchmark.get('provider_id')}"
        )
