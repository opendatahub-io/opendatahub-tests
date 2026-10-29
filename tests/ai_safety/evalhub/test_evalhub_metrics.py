import pytest
import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.route import Route
from ocp_resources.service import Service

from tests.ai_safety.evalhub.constants import (
    EVALHUB_HEALTH_PATH,
    EVALHUB_HTTP_METRICS,
    EVALHUB_OTEL_METRICS_STDOUT_CONFIG,
)
from tests.ai_safety.evalhub.utils import fetch_evalhub_metrics
from utilities.guardrails import get_auth_headers


@pytest.mark.parametrize(
    "model_namespace, evalhub_cr",
    [
        pytest.param(
            {"name": "test-evalhub-metrics"},
            {"otel": EVALHUB_OTEL_METRICS_STDOUT_CONFIG},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.ai_safety
class TestEvalHubMetrics:
    """Tests for the EvalHub Prometheus metrics endpoint."""

    def test_evalhub_metrics_endpoint(
        self,
        current_client_token: str,
        evalhub_ca_bundle_file: str,
        admin_client: DynamicClient,
        evalhub_metrics_service: Service,
    ) -> None:
        """Verify /metrics returns 200 and includes the OTEL HTTP server metrics.

        The metrics endpoint is on the cluster-internal port 8081 with no Route,
        so it is scraped via the service DNS name from inside the EvalHub pod.
        HTTP metrics are only emitted with OTEL metrics enabled on the EvalHub CR.
        """
        body = fetch_evalhub_metrics(
            client=admin_client,
            evalhub_metrics_service=evalhub_metrics_service,
        )
        for metric in EVALHUB_HTTP_METRICS:
            assert metric in body, f"Expected metric '{metric}' not found in /metrics response"

    def test_evalhub_metrics_recorded_for_requests(
        self,
        current_client_token: str,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
        admin_client: DynamicClient,
        evalhub_metrics_service: Service,
    ) -> None:
        """Given: a running EvalHub instance with metrics service.
        When: GET /api/v1/health is called, then /metrics is scraped.
        Then: /metrics contains a request count for the health path.
        """
        headers = get_auth_headers(token=current_client_token)

        # Hit the health endpoint through the Route to generate a metric entry
        health_url = f"https://{evalhub_route.host}{EVALHUB_HEALTH_PATH}"
        health_resp = requests.get(
            url=health_url,
            headers=headers,
            verify=evalhub_ca_bundle_file,
            timeout=10,
        )
        assert health_resp.status_code == 200

        # Scrape metrics from the internal service and verify the health path appears
        metrics = fetch_evalhub_metrics(
            client=admin_client,
            evalhub_metrics_service=evalhub_metrics_service,
        )
        assert EVALHUB_HEALTH_PATH in metrics, f"Expected request count for '{EVALHUB_HEALTH_PATH}' in /metrics output"
