"""OGX telemetry: Prometheus metrics endpoint and OTLP export.

Covers the shipped telemetry design: OGX exposes its OpenTelemetry metrics on
an auth-exempt pull endpoint on a dedicated port (never routed), the operator
creates the ServiceMonitor for it, and cluster monitoring can scrape it. The
pre-existing OTLP push export is verified by pointing it at the mock upstream,
which records the exports it receives.

The Prometheus-integration test additionally requires user workload monitoring
(a cluster-scoped, cluster-admin setting the test cannot and should not flip)
and is skipped with an actionable message on clusters without it.
"""

import time
from urllib.parse import quote

import httpx
import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod
from ocp_resources.service_monitor import ServiceMonitor
from ogx_client import OgxClient

from tests.ogx.constants import OGX_CLIENT_VERIFY_SSL, OGX_METRICS_PORT
from tests.ogx.mockupstream.constants import MOCK_MAAS_QUALIFIED_MODEL_ID
from tests.ogx.mockupstream.deployments import MockUpstream
from tests.ogx.utils import wait_for_unique_ogx_pod
from utilities.resources.ogx_server import OgxServer

LOGGER = structlog.get_logger(name=__name__)

OTEL_NAMESPACE = {"name": "test-ogx-otel", "randomize_name": True}

UWM_NAMESPACE = "openshift-user-workload-monitoring"
UWM_PROMETHEUS_POD_SELECTOR = "app.kubernetes.io/name=prometheus"

# Metric families asserted after inference traffic (subset of the shipped
# ogx.* instrumentation).
EXPECTED_METRIC = "ogx_requests_total"

# OTLP http/protobuf metrics are posted to {endpoint}/v1/metrics.
OTLP_METRICS_PATH = "/v1/metrics"
# The OTLP export interval is shortened to 5s in the fixture; allow a margin
# for the first periodic export to happen after the inference traffic.
OTLP_EXPORT_WAIT_TIMEOUT = 60
OTLP_EXPORT_POLL_INTERVAL = 5

# User workload Prometheus scrapes on a ~30s interval; a freshly created OGX
# target needs a few scrape cycles before its series are queryable.
PROMETHEUS_SCRAPE_WAIT_TIMEOUT = 300
PROMETHEUS_POLL_INTERVAL = 30


def _send_chat_completion(ogx_client: OgxClient) -> None:
    response = ogx_client.chat.completions.create(
        model=MOCK_MAAS_QUALIFIED_MODEL_ID,
        messages=[{"role": "user", "content": "otel metrics traffic"}],
    )
    assert response.choices[0].message.content, "Empty completion from mock model"


def _get_ogx_pod(admin_client: DynamicClient, namespace: str) -> Pod:
    return wait_for_unique_ogx_pod(client=admin_client, namespace=namespace)


def _scrape_metrics_from_pod(pod: Pod) -> str:
    """Fetch /metrics from inside the OGX pod (the port is network-isolated)."""
    return pod.execute(
        command=["wget", "-qO-", f"http://localhost:{OGX_METRICS_PORT}/metrics"],
        timeout=60,
    )


def _query_user_workload_prometheus(prometheus_pod: Pod, query: str) -> str:
    """Run an instant PromQL query inside the user workload Prometheus pod."""
    query_url = f"http://localhost:9090/api/v1/query?query={quote(query)}"
    return prometheus_pod.execute(command=["wget", "-qO-", query_url], timeout=120)


@pytest.mark.ogx
@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [pytest.param(OTEL_NAMESPACE)],
    indirect=True,
)
class TestOtelMetrics:
    """Prometheus metrics endpoint and OTLP export."""

    @pytest.mark.tier2
    def test_metrics_endpoint_exposes_ogx_series(
        self, admin_client: DynamicClient, ogx_client: OgxClient, ogx_server: OgxServer
    ) -> None:
        """Verify the auth-exempt metrics endpoint serves ogx.* series.

        Given: an OGXServer deployed with spec.monitoring enabled (metrics
            port 9464).
        When: inference traffic is sent and the metrics endpoint is scraped
            from inside the OGX pod (the port is never exposed via Route).
        Then: the endpoint answers with Prometheus exposition format
            containing OGX's request metrics.
        """
        _send_chat_completion(ogx_client=ogx_client)

        pod = _get_ogx_pod(admin_client=admin_client, namespace=ogx_server.namespace)
        metrics_output = _scrape_metrics_from_pod(pod=pod)
        assert EXPECTED_METRIC in metrics_output, (
            f"{EXPECTED_METRIC} missing from /metrics output; first 500 chars: {metrics_output[:500]}"
        )

    @pytest.mark.tier2
    def test_operator_creates_service_monitor(self, admin_client: DynamicClient, ogx_server: OgxServer) -> None:
        """Verify the operator creates a ServiceMonitor for the metrics port.

        Given: an OGXServer deployed with spec.monitoring enabled.
        Then: a ServiceMonitor named after the server exists in the namespace,
            targeting the metrics port so cluster monitoring can scrape it.
        """
        service_monitor = ServiceMonitor(
            client=admin_client,
            name=f"{ogx_server.name}-service-monitor",
            namespace=ogx_server.namespace,
        )
        assert service_monitor.exists, f"ServiceMonitor '{service_monitor.name}' not found in {ogx_server.namespace}"
        ports = [
            port
            for endpoint in service_monitor.instance.spec.endpoints
            for port in [endpoint.get("port", endpoint.get("targetPort"))]
        ]
        assert ports, f"ServiceMonitor has no endpoints: {service_monitor.instance.spec}"
        # The operator targets the service port named "metrics", which maps to
        # the metrics port configured in spec.monitoring (9464).
        assert "metrics" in ports, f"ServiceMonitor does not target the metrics port (expected 'metrics', got: {ports})"
        LOGGER.info("ServiceMonitor endpoints", name=service_monitor.name, ports=ports)

    @pytest.mark.tier2
    def test_otlp_export_reaches_configured_endpoint(
        self, ogx_client: OgxClient, mock_maas_upstream: MockUpstream, ogx_server: OgxServer
    ) -> None:
        """Verify the OTLP push export posts metrics to the configured endpoint.

        Given: an OGXServer whose OTEL_EXPORTER_OTLP_ENDPOINT points at the
            mock upstream (with a short export interval).
        When: inference traffic is sent.
        Then: the mock receives OTLP metric exports on the OTLP metrics path.
        """
        _send_chat_completion(ogx_client=ogx_client)

        # The export is periodic (5s interval in the fixture): the first
        # export after the traffic may not have happened yet when the debug
        # endpoint is first queried, so poll for it.
        deadline = time.monotonic() + OTLP_EXPORT_WAIT_TIMEOUT
        otlp_posts: list = []
        recorded_paths: list = []
        while True:
            with httpx.Client(
                verify=OGX_CLIENT_VERIFY_SSL,
                timeout=60,
                headers=mock_maas_upstream.debug_headers(),
            ) as client:
                response = client.get(f"{mock_maas_upstream.route_url}/v1/debug/requests")
                response.raise_for_status()
                recorded = response.json()["data"]
            otlp_posts = [req for req in recorded if req["path"].endswith(OTLP_METRICS_PATH)]
            if otlp_posts or time.monotonic() >= deadline:
                break
            recorded_paths = [req["path"] for req in recorded]
            time.sleep(OTLP_EXPORT_POLL_INTERVAL)

        assert otlp_posts, (
            f"No OTLP export recorded on {OTLP_METRICS_PATH} within "
            f"{OTLP_EXPORT_WAIT_TIMEOUT}s; recorded paths: {recorded_paths}"
        )

    @pytest.mark.tier2
    def test_user_workload_prometheus_scrapes_ogx_metrics(
        self, admin_client: DynamicClient, ogx_client: OgxClient, ogx_server: OgxServer
    ) -> None:
        """Verify cluster monitoring scrapes the OGX metrics (UWM-gated).

        Given: user workload monitoring enabled on the cluster (cluster-scoped
            setting; skipped with an actionable message when absent) and an
            OGXServer with spec.monitoring enabled.
        When: inference traffic is sent and the user workload Prometheus is
            queried through its API.
        Then: the OGX request metric is present with data, i.e. the scrape
            pipeline (ServiceMonitor -> Prometheus) works end to end.
        """
        prometheus_pods = list(
            Pod.get(
                client=admin_client,
                namespace=UWM_NAMESPACE,
                label_selector=UWM_PROMETHEUS_POD_SELECTOR,
            )
        )
        if not prometheus_pods:
            pytest.skip(
                f"User workload monitoring is not enabled on this cluster "
                f"(no Prometheus pods in {UWM_NAMESPACE}); enable it via the "
                f"cluster-monitoring-config ConfigMap to run this test"
            )

        _send_chat_completion(ogx_client=ogx_client)

        # Restrict the query to this deployment's target: Prometheus labels
        # scraped series with the job "<ogx-server-name>-service" (the
        # ServiceMonitor's target service), so unrelated OGX deployments on
        # the cluster cannot satisfy the assertion.
        job = f"{ogx_server.name}-service"
        query = f'{EXPECTED_METRIC}{{job="{job}"}}'

        prometheus_pod = prometheus_pods[0]
        # A freshly created target needs a few of Prometheus' ~30s scrape
        # cycles before its series are queryable, so poll until the metric
        # has data (or the bounded wait times out).
        deadline = time.monotonic() + PROMETHEUS_SCRAPE_WAIT_TIMEOUT
        has_data = False
        result = ""
        while True:
            result = _query_user_workload_prometheus(prometheus_pod=prometheus_pod, query=query)
            assert '"status":"success"' in result, f"Prometheus query failed: {result[:300]}"
            has_data = '"result":[' in result and '"result":[]' not in result
            if has_data or time.monotonic() >= deadline:
                break
            LOGGER.info("OGX series not queryable in user workload Prometheus yet, waiting", job=job)
            time.sleep(PROMETHEUS_POLL_INTERVAL)

        assert has_data, (
            f"{EXPECTED_METRIC} has no data for job {job!r} in user workload Prometheus "
            f"within {PROMETHEUS_SCRAPE_WAIT_TIMEOUT}s; the OGX target may not be up. "
            f"Query result: {result[:500]}"
        )
