"""Alert firing validation on a dedicated MaaS test cluster."""

from collections.abc import Generator
from time import monotonic
from urllib.parse import urlsplit

import portforward
import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.dsc_initialization import DSCInitialization
from ocp_resources.namespace import Namespace
from ocp_resources.prometheus import Prometheus
from ocp_resources.prometheus_rule import PrometheusRule
from ocp_resources.resource import ResourceEditor
from ocp_resources.service import Service
from ocp_resources.service_monitor import ServiceMonitor
from timeout_sampler import TimeoutSampler

from tests.ai_gateway.image_constants import AIGatewayImages
from tests.ai_gateway.models_as_a_service.observability.authorino_alerts import (
    AuthorinoMetricsClient,
    post_authorino_request,
    query_alert_metrics,
)
from tests.ai_gateway.models_as_a_service.observability.utils import resolve_maas_monitoring_namespace
from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import MAAS_GATEWAY_NAME, MAAS_GATEWAY_NAMESPACE
from utilities.general import generate_random_name
from utilities.resources.llm_inference_service import LLMInferenceService

LOGGER = structlog.get_logger(name=__name__)
ALERT_NAME = "MaaSAuthorinoAuthenticationHighFailureRate"
RULE_NAME = "authorino-maas-authentication-alerts"
POLL_SECONDS = 15
FIRING_TIMEOUT_SECONDS = 720
RECOVERY_TIMEOUT_SECONDS = 420


@pytest.fixture
def authorino_monitoring_stack(admin_client: DynamicClient, dsci_resource: DSCInitialization) -> Generator[str]:
    """Enable DSCI metrics and alerting for the test, restoring prior settings afterward."""
    monitoring = dsci_resource.instance.to_dict().get("spec", {}).get("monitoring", {})
    with ResourceEditor(
        patches={
            dsci_resource: {
                "spec": {
                    "monitoring": {
                        "metrics": monitoring.get("metrics") or {"storage": {"size": "5Gi", "retention": "1d"}},
                        "alerting": monitoring.get("alerting") or {},
                    }
                }
            }
        }
    ):
        dsci_resource.wait_for_condition(condition="MonitoringStackAvailable", status="True", timeout=600)
        namespace = resolve_maas_monitoring_namespace(admin_client=admin_client)
        assert namespace, "DSCI monitoring namespace is not configured"
        # These must come from maas-controller; installing test copies would hide product regressions.
        ServiceMonitor(client=admin_client, name="authorino-server-metrics", namespace=namespace).wait(timeout=300)
        PrometheusRule(client=admin_client, name=RULE_NAME, namespace=namespace, api_group="monitoring.rhobs").wait(
            timeout=300
        )
        yield namespace


@pytest.fixture
def authorino_model_namespace(admin_client: DynamicClient, authorino_monitoring_stack: str) -> Generator[Namespace]:
    """Provide an isolated namespace for the simulator and remove it after the test."""
    with Namespace(client=admin_client, name=generate_random_name(prefix="authorino-alert-")) as namespace:
        yield namespace


@pytest.fixture
def authorino_model_url(admin_client: DynamicClient, authorino_model_namespace: Namespace) -> Generator[str]:
    """Deploy a CPU-only simulator behind the MaaS gateway and clean it up afterward."""
    with LLMInferenceService(
        client=admin_client,
        name="authorino-alert-simulator",
        namespace=authorino_model_namespace.name,
        model={"uri": "hf://placeholder/no-model", "name": "facebook/opt-125m"},
        storage_initializer={"enabled": False},
        replicas=1,
        router={"route": {}, "gateway": {"refs": [{"name": MAAS_GATEWAY_NAME, "namespace": MAAS_GATEWAY_NAMESPACE}]}},
        template={
            "containers": [
                {
                    "name": "main",
                    "image": AIGatewayImages.LLMD_INFERENCE_SIM,
                    "command": ["/app/llm-d-inference-sim"],
                    "args": [
                        "--port",
                        "8000",
                        "--model",
                        "facebook/opt-125m",
                        "--mode",
                        "random",
                        "--no-mm-encoder-only",
                        "--ssl-certfile",
                        "/var/run/kserve/tls/tls.crt",
                        "--ssl-keyfile",
                        "/var/run/kserve/tls/tls.key",
                    ],
                    "ports": [{"name": "https", "containerPort": 8000}],
                    "env": [
                        {"name": "POD_NAME", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}},
                        {"name": "POD_NAMESPACE", "valueFrom": {"fieldRef": {"fieldPath": "metadata.namespace"}}},
                        {"name": "POD_IP", "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}}},
                    ],
                    "resources": {
                        "requests": {"cpu": "100m", "memory": "256Mi"},
                        "limits": {"cpu": "500m", "memory": "512Mi"},
                    },
                    "readinessProbe": {"httpGet": {"path": "/ready", "port": "https", "scheme": "HTTPS"}},
                    "livenessProbe": {"httpGet": {"path": "/health", "port": "https", "scheme": "HTTPS"}},
                }
            ]
        },
    ) as model:
        model.wait_for_condition(condition="Ready", status="True", timeout=600)
        yield f"{model.instance.status.url.rstrip('/')}/v1/chat/completions"


@pytest.fixture
def authorino_product_prometheus(
    admin_client: DynamicClient, kubconfig_filepath: str, authorino_monitoring_stack: str
) -> Generator[AuthorinoMetricsClient]:
    """Connect to product Thanos without filtering out namespace-less aggregated alerts."""
    namespace = authorino_monitoring_stack
    prometheus_resource = Prometheus(
        client=admin_client, name="data-science-monitoringstack", namespace=namespace, api_group="monitoring.rhobs"
    )
    prometheus_resource.wait_for_condition(condition="Available", status="True", timeout=600)
    Deployment(
        client=admin_client, name="thanos-querier-data-science-thanos-querier", namespace=namespace
    ).wait_for_replicas(timeout=300)
    service = Service(
        client=admin_client,
        name="thanos-querier-data-science-thanos-querier",
        namespace=namespace,
    )
    service.wait(timeout=300)
    with portforward.forward(
        namespace=namespace,
        pod_or_service=service.name,
        from_port=0,
        to_port=10902,
        config_path=kubconfig_filepath,
        bind_ip="127.0.0.1",
        waiting=2,
    ) as forwarder:
        prometheus = AuthorinoMetricsClient(
            api_url=f"http://127.0.0.1:{forwarder.from_port}",
            verify_ssl=create_ca_bundle_file(client=admin_client),
            monitoring_namespace=namespace,
        )
        # Pod readiness precedes Thanos discovering the new Prometheus sidecars.
        expected_replicas = prometheus_resource.instance.spec.get("replicas", 1)
        for response in TimeoutSampler(
            wait_timeout=300,
            sleep=10,
            func=requests.get,
            exceptions_dict={requests.exceptions.ConnectionError: [], requests.exceptions.Timeout: []},
            url=f"{prometheus.api_url}/api/v1/stores",
            timeout=30,
        ):
            response.raise_for_status()
            result = response.json()
            assert result.get("status") == "success", "Thanos store discovery failed"
            stores = [
                store
                for store in result["data"].get("sidecar", [])
                if any(
                    labels.get("prometheus") == f"{namespace}/{prometheus_resource.name}"
                    for labels in store.get("labelSets", [])
                )
            ]
            if len(stores) >= expected_replicas and all(not store.get("lastError") for store in stores):
                break
        for samples in TimeoutSampler(
            wait_timeout=300,
            sleep=10,
            func=query_alert_metrics,
            exceptions_dict={requests.exceptions.ConnectionError: [], requests.exceptions.Timeout: []},
            prometheus=prometheus,
            query=(
                'min(up{namespace=~"kuadrant-system|rh-connectivity-link",'
                'exported_service="authorino-controller-metrics"})'
            ),
        ):
            if samples and all(sample["value"][1] == "1" for sample in samples):
                break
        yield prometheus


@pytest.mark.tier3
@pytest.mark.metrics
@pytest.mark.slow
def test_authorino_authentication_alert_fires(
    admin_client: DynamicClient,
    authorino_product_prometheus: AuthorinoMetricsClient,
    authorino_model_url: str,
) -> None:
    """Given an inactive shipped alert and a protected model in a test namespace,
    when sustained unauthenticated requests exceed its threshold,
    then the alert transitions through pending to firing and clears after traffic stops.
    """
    prometheus = authorino_product_prometheus
    target_url = authorino_model_url
    target = urlsplit(url=target_url)
    assert target.scheme == "https" and target.hostname, "Simulator must expose an HTTPS model URL"
    assert not (target.username or target.password or target.query or target.fragment), "Use a URL without credentials"
    rule_resource = PrometheusRule(
        client=admin_client,
        name=RULE_NAME,
        namespace=prometheus.monitoring_namespace,
        api_group="monitoring.rhobs",
    )
    assert rule_resource.exists, (
        "Authorino authentication rule is missing from DSCI monitoring (monitoring.rhobs); "
        "use a maas-controller build with Authorino monitoring enabled"
    )
    alert_rules = [
        rule
        for group in rule_resource.instance.to_dict()["spec"]["groups"]
        for rule in group["rules"]
        if rule.get("alert") == ALERT_NAME
    ]
    assert len(alert_rules) == 1, "Shipped authentication failure alert is missing or duplicated"
    assert alert_rules[0].get("for") == "5m", "Review test timing if the shipped hold duration changes"
    assert "auth_server_authconfig_response_status_total" in alert_rules[0]["expr"], (
        "Authorino rule must use the OTEL-normalized authentication counter"
    )
    namespaces = [
        namespace
        for namespace in ("kuadrant-system", "rh-connectivity-link")
        if Deployment(client=admin_client, name="authorino", namespace=namespace).exists
    ]
    assert len(namespaces) == 1, "Expected one Authorino deployment in a supported policy-engine namespace"
    namespace = namespaces[0]
    alert_query = f'ALERTS{{alertname="{ALERT_NAME}"}}'
    target_up_query = f'min(up{{namespace="{namespace}",exported_service="authorino-controller-metrics"}})'
    target_up = query_alert_metrics(prometheus=prometheus, query=target_up_query)
    assert target_up and target_up[0]["value"][1] == "1", "Authorino scrape target is not healthy"
    source_query = f'auth_server_authconfig_response_status_total{{namespace="{namespace}"}}'
    failures_selector = (
        f'auth_server_authconfig_response_status_total{{namespace="{namespace}",status="UNAUTHENTICATED"}}'
    )
    assert query_alert_metrics(prometheus=prometheus, query=source_query), (
        "Authorino authentication metrics are missing"
    )
    assert not query_alert_metrics(prometheus=prometheus, query=alert_query), "Alert is already pending or firing"
    pending_seen = False
    firing_seen = False
    traffic_started = False
    request_count = 0
    started = monotonic()
    next_poll = started
    try:
        with requests.Session() as session:
            session.trust_env = False
            session.verify = prometheus.verify_ssl
            # No credentials: reject at gateway authentication before model inference.
            traffic_started = True
            for response in TimeoutSampler(
                wait_timeout=FIRING_TIMEOUT_SECONDS,
                sleep=2,
                func=post_authorino_request,
                exceptions_dict={requests.exceptions.ConnectionError: [], requests.exceptions.Timeout: []},
                session=session,
                url=target_url,
            ):
                assert response.status_code == 401, (
                    f"Expected authentication rejection, got HTTP {response.status_code}"
                )
                request_count += 1
                if monotonic() >= next_poll:
                    try:
                        samples = query_alert_metrics(prometheus=prometheus, query=alert_query)
                    except requests.exceptions.ConnectionError, requests.exceptions.Timeout:
                        # Retry on the next iteration within the existing firing deadline.
                        continue
                    states = {sample["metric"]["alertstate"] for sample in samples}
                    pending_seen |= "pending" in states
                    LOGGER.info(
                        "Authorino alert observation",
                        states=sorted(states),
                        requests=request_count,
                        elapsed_seconds=round(monotonic() - started),
                    )
                    if "firing" in states:
                        firing_seen = True
                        assert monotonic() - started >= 300, "Alert fired before its five-minute hold duration"
                        break
                    next_poll = monotonic() + POLL_SECONDS
        assert pending_seen, "Never observed the shipped alert pending"
        assert firing_seen, "Alert did not fire within the hold window plus evaluation/scrape allowance"
        # Raw totals can drop when OTEL restarts or old exported series go stale.
        traffic_seconds = max(1, int(monotonic() - started))
        failure_increase = query_alert_metrics(
            prometheus=prometheus, query=f"sum(increase({failures_selector}[{traffic_seconds}s]))"
        )
        assert sum(float(sample["value"][1]) for sample in failure_increase) > 0, (
            "HTTP rejections did not increase Authorino UNAUTHENTICATED metrics"
        )
    finally:
        # No worker remains running on assertion failure; allow the five-minute rate window to drain.
        if traffic_started:
            for failure_rate in TimeoutSampler(
                wait_timeout=RECOVERY_TIMEOUT_SECONDS,
                sleep=POLL_SECONDS,
                func=query_alert_metrics,
                exceptions_dict={requests.exceptions.ConnectionError: [], requests.exceptions.Timeout: []},
                prometheus=prometheus,
                query=f"sum(rate({failures_selector}[5m]))",
            ):
                try:
                    alerts = query_alert_metrics(prometheus=prometheus, query=alert_query)
                    target_up = query_alert_metrics(prometheus=prometheus, query=target_up_query)
                except requests.exceptions.ConnectionError, requests.exceptions.Timeout:
                    # The sampler also bounds retries for queries made after it yields.
                    continue
                assert target_up and target_up[0]["value"][1] == "1", "Authorino scrape target failed during recovery"
                if not alerts and all(float(sample["value"][1]) <= 0 for sample in failure_rate):
                    LOGGER.info("Authorino failure traffic stopped and alert cleared")
                    break
