import pytest
import requests
from ocp_resources.route import Route

from tests.model_explainability.evalhub.constants import (
    EVALHUB_HEALTH_PATH,
    EVALHUB_METRICS_PATH,
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
@pytest.mark.model_explainability
class TestEvalHubMetrics:
    """Tests for the EvalHub Prometheus metrics endpoint."""

    def test_evalhub_metrics_endpoint(
        self,
        current_client_token: str,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
    ) -> None:
        """Verify /metrics returns 200 and includes expected Prometheus metrics."""
        url = f"https://{evalhub_route.host}{EVALHUB_METRICS_PATH}"
        headers = get_auth_headers(token=current_client_token)
        response = requests.get(
            url=url,
            headers=headers,
            verify=evalhub_ca_bundle_file,
            timeout=10,
        )
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
    ) -> None:
        """After hitting /api/v1/health, /metrics should show a request count for that path."""
        headers = get_auth_headers(token=current_client_token)

        # Hit the health endpoint to generate a metric entry
        health_url = f"https://{evalhub_route.host}{EVALHUB_HEALTH_PATH}"
        health_resp = requests.get(
            url=health_url,
            headers=headers,
            verify=evalhub_ca_bundle_file,
            timeout=10,
        )
        assert health_resp.status_code == 200

        # Scrape metrics and verify the health path appears
        metrics_url = f"https://{evalhub_route.host}{EVALHUB_METRICS_PATH}"
        metrics_resp = requests.get(
            url=metrics_url,
            headers=headers,
            verify=evalhub_ca_bundle_file,
            timeout=10,
        )
        assert metrics_resp.status_code == 200
        assert EVALHUB_HEALTH_PATH in metrics_resp.text, (
            f"Expected request count for '{EVALHUB_HEALTH_PATH}' in /metrics output"
        )
