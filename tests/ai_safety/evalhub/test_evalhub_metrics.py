import pytest
import requests
from ocp_resources.route import Route
from ocp_resources.service import Service

from tests.ai_safety.evalhub.constants import (
    EVALHUB_HEALTH_PATH,
    EVALHUB_METRICS_PATH,
    EVALHUB_METRICS_PORT,
)
from utilities.guardrails import get_auth_headers


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-metrics"},
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
        evalhub_metrics_service: Service,
    ) -> None:
        """Verify /metrics returns 200 and includes expected Prometheus metrics.

        The metrics endpoint is on the cluster-internal port 8081 with no Route,
        so it is accessed via the service DNS name rather than the API Route.
        """
        url = (
            f"http://{evalhub_metrics_service.name}"
            f".{evalhub_metrics_service.namespace}"
            f".svc.cluster.local:{EVALHUB_METRICS_PORT}{EVALHUB_METRICS_PATH}"
        )
        response = requests.get(url=url, timeout=10)
        assert response.status_code == 200, f"Expected 200 from /metrics, got {response.status_code}"
        body = response.text
        for metric in (
            "http_requests_total",
            "http_request_duration_seconds",
            "http_requests_in_flight",
        ):
            assert metric in body, f"Expected metric '{metric}' not found in /metrics response"

    def test_evalhub_metrics_recorded_for_requests(
        self,
        current_client_token: str,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
        evalhub_metrics_service: Service,
    ) -> None:
        """After hitting /api/v1/health, /metrics should show a request count for that path.

        Health is hit via the Route (auth required); metrics are scraped from the
        cluster-internal metrics service (no auth, no Route).
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
        metrics_url = (
            f"http://{evalhub_metrics_service.name}"
            f".{evalhub_metrics_service.namespace}"
            f".svc.cluster.local:{EVALHUB_METRICS_PORT}{EVALHUB_METRICS_PATH}"
        )
        metrics_resp = requests.get(url=metrics_url, timeout=10)
        assert metrics_resp.status_code == 200
        assert EVALHUB_HEALTH_PATH in metrics_resp.text, (
            f"Expected request count for '{EVALHUB_HEALTH_PATH}' in /metrics output"
        )
