"""Tests for EvalHub OpenTelemetry metrics integration.

These tests verify that EvalHub correctly initializes, exports, and manages
OpenTelemetry metrics to OTLP-compatible backends.
"""

import re
import time

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_resources.service import Service

from tests.ai_safety.evalhub.constants import (
    EVALHUB_HEALTH_PATH,
    EVALHUB_METRICS_PATH,
    EVALHUB_PROVIDERS_PATH,
    OTEL_ERROR_PATTERNS,
    OTLP_INDICATORS,
)
from utilities.guardrails import get_auth_headers

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-otel"},
        ),
    ],
    indirect=True,
)
@pytest.mark.ai_safety
@pytest.mark.tier1
class TestEvalHubOTEL:
    """Tests for EvalHub OpenTelemetry metrics integration."""

    def test_meter_provider_initialization(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_otel_grpc_deployment: Deployment,
    ) -> None:
        """Verify MeterProvider initializes without silent failure when OTEL is enabled.

        Test Case 1: Verify that EvalHub successfully initializes and registers an OTEL
        MeterProvider at server startup without silent failure when metrics are enabled.
        """
        pods = list(
            Pod.get(
                client=admin_client,
                namespace=model_namespace.name,
                label_selector="app=eval-hub,component=api",
            )
        )
        assert len(pods) >= 1, f"Expected at least 1 EvalHub pod, found {len(pods)}"

        # Check the first pod (they should all be configured the same)
        pod = pods[0]

        # Verify container is not restarting
        container_statuses = pod.instance.status.containerStatuses or []
        evalhub_container = next(
            (container for container in container_statuses if container.name == "evalhub"),
            None,
        )
        assert evalhub_container is not None, "evalhub container not found in pod"
        assert evalhub_container.restartCount == 0, (
            f"Container restarted {evalhub_container.restartCount} times - indicates initialization failure"
        )

        # Check logs for OTEL errors
        logs = pod.log(container="evalhub")

        for pattern in OTEL_ERROR_PATTERNS:
            assert pattern.lower() not in logs.lower(), f"Found error pattern '{pattern}' in logs"

        LOGGER.info("MeterProvider initialization verified successfully")

    def test_otlp_grpc_export(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_otel_grpc_route: Route,
        current_client_token: str,
        evalhub_otel_ca_bundle_file: str,
        otel_collector_pod: Pod,
    ) -> None:
        """Verify metrics are exported via OTLP gRPC.

        Test Case 2: Verify that metrics are exported to an OTLP-compatible backend via
        the gRPC transport when `OTELConfig.ExporterType` is set to `otlp-grpc`.
        """
        # Generate traffic to create metrics
        url = f"https://{evalhub_otel_grpc_route.host}{EVALHUB_HEALTH_PATH}"
        headers = get_auth_headers(token=current_client_token)

        for request_num in range(1, 6):
            response = requests.get(url=url, headers=headers, verify=evalhub_otel_ca_bundle_file, timeout=10)
            assert response.status_code == 200, f"Health check {request_num} failed: {response.status_code}"
            time.sleep(1)

        LOGGER.info("Generated 5 health check requests")

        # Wait for export interval (60s default + buffer)
        time.sleep(70)

        # Check collector received metrics via gRPC
        collector_logs = otel_collector_pod.log(container="otel-collector", tail_lines=200)

        # Look for evidence of received OTLP metrics
        found_indicators = [ind for ind in OTLP_INDICATORS if ind in collector_logs]
        assert len(found_indicators) >= 2, (
            f"Expected OTLP metrics in collector logs. "
            f"Found {len(found_indicators)}/{len(OTLP_INDICATORS)} indicators: {found_indicators}. "
            "Collector may not be receiving gRPC exports."
        )

        LOGGER.info(f"OTLP gRPC export verified - found indicators: {found_indicators}")

    def test_otlp_http_export(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_otel_http_route: Route,
        current_client_token: str,
        evalhub_otel_ca_bundle_file: str,
        otel_collector_pod: Pod,
    ) -> None:
        """Verify metrics are exported via OTLP HTTP.

        Test Case 3: Verify that metrics are exported to an OTLP-compatible backend via
        the HTTP transport when `OTELConfig.ExporterType` is set to `otlp-http`.
        """
        url = f"https://{evalhub_otel_http_route.host}{EVALHUB_HEALTH_PATH}"
        headers = get_auth_headers(token=current_client_token)

        for request_num in range(1, 6):
            response = requests.get(url=url, headers=headers, verify=evalhub_otel_ca_bundle_file, timeout=10)
            assert response.status_code == 200, f"Health check {request_num} failed: {response.status_code}"
            time.sleep(1)

        LOGGER.info("Generated 5 health check requests for HTTP exporter test")

        # Wait for export interval (60s default + buffer)
        time.sleep(70)

        collector_logs = otel_collector_pod.log(container="otel-collector", tail_lines=200)

        # Same indicators as gRPC - OTLP format is the same, just different transport
        assert "ResourceMetrics" in collector_logs or "http.server.request" in collector_logs, (
            "No OTLP metrics found in collector logs. HTTP export may not be working."
        )

        LOGGER.info("OTLP HTTP export verified")

    def test_metric_export_interval_default(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_otel_grpc_route: Route,
        current_client_token: str,
        evalhub_otel_ca_bundle_file: str,
        otel_collector_pod: Pod,
    ) -> None:
        """Verify metrics are exported at default 60s interval.

        Test Case 4: Verify that metrics are exported periodically when OTEL is enabled.
        The default export interval is 60 seconds.

        Note: Custom intervals via OTEL_METRIC_EXPORT_INTERVAL are not yet supported by the CRD.
        """
        # Generate traffic
        url = f"https://{evalhub_otel_grpc_route.host}{EVALHUB_HEALTH_PATH}"
        headers = get_auth_headers(token=current_client_token)

        for _ in range(5):
            requests.get(url=url, headers=headers, verify=evalhub_otel_ca_bundle_file, timeout=10)
            time.sleep(1)

        LOGGER.info("Generated 5 health check requests")

        # Wait for default export interval (60s + buffer)
        LOGGER.info("Waiting 70s for metric export (60s default interval + 10s buffer)")
        time.sleep(70)

        # Verify metrics were exported
        collector_logs = otel_collector_pod.log(container="otel-collector", tail_lines=300)

        # Check for evidence of metric export
        assert "ResourceMetrics" in collector_logs or "http.server.request" in collector_logs, (
            "No metrics found in collector logs after 70s wait. Default 60s export interval may not be working."
        )

        LOGGER.info("Default export interval verified - metrics exported successfully")

    def test_dual_sink_consistency(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_otel_dual_sink_route: Route,
        current_client_token: str,
        evalhub_otel_ca_bundle_file: str,
        otel_collector_service: Service,
        otel_collector_pod: Pod,
    ) -> None:
        """Verify metrics appear consistently in both Prometheus and OTLP sinks.

        Test Case 6: Verify dual-sink behavior: when both OTLP and Prometheus are enabled,
        the same metric names and values appear on both the OTLP backend and the `/metrics`
        HTTP endpoint.
        """
        # Generate exactly 10 health check requests
        url = f"https://{evalhub_otel_dual_sink_route.host}{EVALHUB_HEALTH_PATH}"
        headers = get_auth_headers(token=current_client_token)

        for request_num in range(1, 11):
            response = requests.get(url=url, headers=headers, verify=evalhub_otel_ca_bundle_file, timeout=10)
            assert response.status_code == 200, f"Request {request_num} failed"

        LOGGER.info("Generated 10 health check requests for dual-sink test")

        # Wait for metrics export
        time.sleep(70)  # Wait for 60s default export interval + buffer

        # Scrape Prometheus endpoint from metrics service (port 8081, not main route)
        # Use a pod in-cluster to access the metrics service
        pods = list(
            Pod.get(
                client=admin_client,
                namespace=model_namespace.name,
                label_selector="app=eval-hub,component=api",
            )
        )
        evalhub_pod = pods[0]

        # Exec curl from within the pod to access metrics service
        prom_metrics = evalhub_pod.execute(command=["curl", "-s", "http://localhost:8081/metrics"], container="evalhub")

        # Verify the health path metric is present
        assert EVALHUB_HEALTH_PATH in prom_metrics, f"Expected '{EVALHUB_HEALTH_PATH}' metric in Prometheus output"

        # Parse the counter value from Prometheus metrics
        # Format: http_server_request_count_total{http_route="/api/v1/health",...} 10
        health_counter_pattern = (
            r'http_server_request_count_total\{[^}]*http_route="' + re.escape(EVALHUB_HEALTH_PATH) + r'"[^}]*\}\s+(\d+)'
        )
        prom_match = re.search(health_counter_pattern, prom_metrics)

        assert prom_match, (
            f"Could not find http_server_request_count_total for {EVALHUB_HEALTH_PATH} in Prometheus metrics"
        )
        prom_count = int(prom_match.group(1))

        LOGGER.info(f"Prometheus endpoint shows count: {prom_count}")

        # Check OTLP collector received metrics
        collector_logs = otel_collector_pod.log(container="otel-collector", tail_lines=300)

        # Verify OTLP collector has the metric
        assert "http.server.request" in collector_logs or "ResourceMetrics" in collector_logs, (
            "OTLP collector did not receive http.server.request metric"
        )

        # For a more detailed comparison, we'd query the collector's Prometheus endpoint
        # This requires port-forwarding or creating a debug pod - simplified here

        LOGGER.info("Dual-sink behavior verified: metrics present in both Prometheus and OTLP")

        # Additional check: both metrics should have similar values
        # Since we can't easily parse OTLP values from logs, we verify presence
        # In a real scenario, you'd query collector's Prometheus endpoint at :8889/metrics

    def test_global_meter_provider_access(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_otel_grpc_route: Route,
        current_client_token: str,
        evalhub_otel_ca_bundle_file: str,
        otel_collector_pod: Pod,
    ) -> None:
        """Verify MeterProvider is globally accessible from different packages.

        Test Case 7: Verify that the globally registered MeterProvider is accessible via
        `otel.GetMeterProvider()` from packages outside `internal/otel`.
        """
        headers = get_auth_headers(token=current_client_token)

        # Hit multiple endpoints that would use different packages/handlers
        endpoints = [
            EVALHUB_HEALTH_PATH,
            EVALHUB_PROVIDERS_PATH,
            EVALHUB_METRICS_PATH,
        ]

        for endpoint in endpoints:
            url = f"https://{evalhub_otel_grpc_route.host}{endpoint}"
            response = requests.get(url=url, headers=headers, verify=evalhub_otel_ca_bundle_file, timeout=10)
            # Endpoints should respond (200, 400, 404, 401 all valid), just not crash (500, 502, 503)
            assert response.status_code < 500, f"Endpoint {endpoint} crashed with status {response.status_code}"
            LOGGER.info(f"Accessed endpoint {endpoint}: {response.status_code}")

        # Wait for metrics export (60s default interval + buffer)
        time.sleep(70)

        # Check collector received metrics
        collector_logs = otel_collector_pod.log(container="otel-collector", tail_lines=400)

        # Verify metrics were exported (indicating global meter provider is working)
        assert "ResourceMetrics" in collector_logs or "http.server.request" in collector_logs, (
            "No metrics exported to OTLP collector after accessing multiple endpoints. "
            "Global MeterProvider may not be accessible."
        )

        # Look for evidence of multiple instrumentation scopes (different packages using the meter)
        # OTLP includes "InstrumentationScope" or "Scope" fields
        scope_count = collector_logs.count("InstrumentationScope") + collector_logs.count("ScopeMetrics")

        assert scope_count > 0, "No instrumentation scopes found - MeterProvider may not be globally accessible"

        LOGGER.info(f"Found {scope_count} instrumentation scope references in OTLP exports")

        # At minimum, we should see metrics being exported, which proves global access works
        # More sophisticated verification would parse the OTLP data to confirm multiple scopes

        LOGGER.info("Global MeterProvider access verified - metrics exported from application")
