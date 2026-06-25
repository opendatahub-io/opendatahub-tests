import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.evalhub import EvalHub
from ocp_resources.namespace import Namespace
from ocp_resources.service import Service
from ocp_resources.service_monitor import ServiceMonitor
from ocp_utilities.monitoring import Prometheus
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_safety.evalhub.constants import (
    EVALHUB_APP_LABEL,
    EVALHUB_COMPONENT_LABEL,
    EVALHUB_METRICS_COMPONENT_LABEL,
    EVALHUB_METRICS_PORT,
    EVALHUB_METRICS_SERVICE_SUFFIX,
    EVALHUB_SCRAPE_INTERVAL,
)
from utilities.monitoring import validate_metrics_field


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-servicemonitor"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.ai_safety
class TestEvalHubServiceMonitor:
    """Tests for the EvalHub ServiceMonitor and metrics Service resources."""

    def test_servicemonitor_exists(
        self,
        evalhub_cr: EvalHub,
        evalhub_service_monitor: ServiceMonitor,
    ) -> None:
        """Verify the operator created a ServiceMonitor with correct owner reference."""
        assert evalhub_service_monitor.exists, (
            f"ServiceMonitor '{evalhub_service_monitor.name}' not found in namespace "
            f"'{evalhub_service_monitor.namespace}'"
        )

        owner_refs = evalhub_service_monitor.instance.metadata.ownerReferences
        assert owner_refs, "ServiceMonitor has no ownerReferences"

        cr_uid = evalhub_cr.instance.metadata.uid
        owner_uids = [ref.uid for ref in owner_refs]
        assert cr_uid in owner_uids, (
            f"Expected ownerReference UID '{cr_uid}' (EvalHub CR) not found in "
            f"ServiceMonitor ownerReferences: {owner_uids}"
        )

    def test_servicemonitor_spec(
        self,
        model_namespace: Namespace,
        evalhub_cr: EvalHub,
        evalhub_service_monitor: ServiceMonitor,
    ) -> None:
        """Validate the ServiceMonitor spec matches the operator's expected configuration."""
        spec = evalhub_service_monitor.instance.spec

        endpoints = spec.endpoints
        assert len(endpoints) == 1, f"Expected 1 endpoint, found {len(endpoints)}"

        ep = endpoints[0]
        assert ep.path == "/metrics", f"Expected endpoint path '/metrics', got '{ep.path}'"
        assert ep.port == "metrics", f"Expected endpoint port 'metrics', got '{ep.port}'"
        assert ep.scheme == "http", f"Expected scheme 'http', got '{ep.scheme}'"
        assert str(ep.interval) == EVALHUB_SCRAPE_INTERVAL, (
            f"Expected scrape interval '{EVALHUB_SCRAPE_INTERVAL}', got '{ep.interval}'"
        )
        assert ep.honorLabels is True, f"Expected honorLabels=true, got {ep.honorLabels}"

        expected_labels = {
            "app": EVALHUB_APP_LABEL,
            "instance": evalhub_cr.name,
            "component": EVALHUB_METRICS_COMPONENT_LABEL,
        }
        selector_labels = dict(spec.selector.matchLabels)
        assert selector_labels == expected_labels, (
            f"Expected selector matchLabels {expected_labels}, got {selector_labels}"
        )

        ns_selector = spec.namespaceSelector
        assert model_namespace.name in ns_selector.matchNames, (
            f"Expected namespace '{model_namespace.name}' in namespaceSelector.matchNames, got {ns_selector.matchNames}"
        )

    def test_metrics_service_exists(
        self,
        evalhub_cr: EvalHub,
        evalhub_metrics_service: Service,
    ) -> None:
        """Verify the metrics Service exists with correct spec."""
        assert evalhub_metrics_service.exists, f"Metrics Service '{evalhub_metrics_service.name}' not found"

        spec = evalhub_metrics_service.instance.spec
        assert spec.type == "ClusterIP", f"Expected Service type 'ClusterIP', got '{spec.type}'"

        ports = spec.ports
        assert len(ports) >= 1, "Expected at least 1 port on the metrics Service"

        metrics_port = next((p for p in ports if p.name == "metrics"), None)
        assert metrics_port is not None, f"Expected port named 'metrics', found: {[p.name for p in ports]}"
        assert metrics_port.port == EVALHUB_METRICS_PORT, (
            f"Expected port {EVALHUB_METRICS_PORT}, got {metrics_port.port}"
        )

        selector = dict(spec.selector)
        assert selector.get("app") == EVALHUB_APP_LABEL, (
            f"Expected selector app='{EVALHUB_APP_LABEL}', got '{selector.get('app')}'"
        )
        assert selector.get("instance") == evalhub_cr.name, (
            f"Expected selector instance='{evalhub_cr.name}', got '{selector.get('instance')}'"
        )
        assert selector.get("component") == EVALHUB_COMPONENT_LABEL, (
            f"Expected selector component='{EVALHUB_COMPONENT_LABEL}', got '{selector.get('component')}'"
        )

    def test_prometheus_target_up(
        self,
        evalhub_cr: EvalHub,
        evalhub_service_monitor: ServiceMonitor,
        prometheus: Prometheus,
    ) -> None:
        """Verify EvalHub appears as an active scrape target (UP) in Prometheus."""
        sm_name = f"{evalhub_cr.name}{EVALHUB_METRICS_SERVICE_SUFFIX}"
        validate_metrics_field(
            prometheus=prometheus,
            metrics_query=f'up{{job="{sm_name}"}}',
            expected_value="1",
            timeout=120,
        )

    def test_metrics_queryable_via_thanos(
        self,
        evalhub_cr: EvalHub,
        evalhub_service_monitor: ServiceMonitor,
        prometheus: Prometheus,
    ) -> None:
        """Verify EvalHub metrics are queryable through the Thanos Querier."""
        sm_name = f"{evalhub_cr.name}{EVALHUB_METRICS_SERVICE_SUFFIX}"
        validate_metrics_field(
            prometheus=prometheus,
            metrics_query=f'http_requests_total{{job="{sm_name}"}}',
            expected_value="0",
            greater_than=True,
            timeout=120,
        )


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-sm-cleanup"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier2
@pytest.mark.ai_safety
class TestEvalHubServiceMonitorCleanup:
    """Tests for ServiceMonitor garbage collection on EvalHub CR deletion."""

    def test_servicemonitor_cleanup(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
    ) -> None:
        """Verify ServiceMonitor is garbage-collected when EvalHub CR is deleted."""
        with EvalHub(
            client=admin_client,
            name="evalhub-cleanup",
            namespace=model_namespace.name,
            database={"type": "sqlite"},
            wait_for_resource=True,
        ) as evalhub:
            deployment = Deployment(
                client=admin_client,
                name=evalhub.name,
                namespace=model_namespace.name,
            )
            deployment.wait_for_replicas(timeout=300)

            sm_name = f"{evalhub.name}{EVALHUB_METRICS_SERVICE_SUFFIX}"
            sm = ServiceMonitor(
                client=admin_client,
                name=sm_name,
                namespace=model_namespace.name,
                ensure_exists=True,
            )
            assert sm.exists, f"ServiceMonitor '{sm_name}' should exist before CR deletion"

        try:
            for sample in TimeoutSampler(
                wait_timeout=60,
                sleep=5,
                func=lambda: (
                    not ServiceMonitor(
                        client=admin_client,
                        name=sm_name,
                        namespace=model_namespace.name,
                    ).exists
                ),
            ):
                if sample:
                    return
        except TimeoutExpiredError:
            pytest.fail(f"ServiceMonitor '{sm_name}' was not garbage-collected within 60s after EvalHub CR deletion")
