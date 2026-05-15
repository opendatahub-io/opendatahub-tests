"""Core EvalHub ServiceMonitor lifecycle tests.

Test Plan Reference: RHAISTRAT-1507
Test Cases: TC-SM-001, TC-SM-002, TC-SM-003, TC-SM-004, TC-SM-005, TC-CFG-001, TC-CFG-002
"""

import time

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.evalhub import EvalHub
from ocp_resources.namespace import Namespace
from ocp_resources.service import Service
from ocp_resources.service_monitor import ServiceMonitor
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_explainability.evalhub.constants import EVALHUB_APP_LABEL
from utilities.constants import Timeout

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-servicemonitor-lifecycle"},
        ),
    ],
    indirect=True,
)
@pytest.mark.smoke
@pytest.mark.model_explainability
class TestEvalHubServiceMonitorLifecycle:
    """Tests for EvalHub ServiceMonitor lifecycle management (RHAISTRAT-1507)."""

    def test_servicemonitor_auto_created_with_metrics_enabled(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_cr: EvalHub,
        evalhub_deployment: Deployment,
    ) -> None:
        """TC-SM-001: Verify ServiceMonitor auto-created when EvalHub deployed with metrics enabled.

        The TrustyAI operator creates a ServiceMonitor resource when an EvalHub CR
        is deployed with the default configuration (metrics enabled). The ServiceMonitor
        targets the HTTPS metrics endpoint with a 30-second scrape interval.
        """
        service_monitors = list(
            ServiceMonitor.get(
                client=admin_client,
                namespace=model_namespace.name,
                label_selector=f"app={EVALHUB_APP_LABEL}",
            )
        )
        assert service_monitors, (
            f"No ServiceMonitor found in namespace '{model_namespace.name}' with label app={EVALHUB_APP_LABEL}. "
            f"Expected operator to auto-create ServiceMonitor when EvalHub CR deployed with metrics enabled."
        )

        service_monitor = service_monitors[0]
        endpoints = service_monitor.instance.spec.endpoints
        assert endpoints, f"ServiceMonitor '{service_monitor.name}' has no endpoints defined"

        endpoint = endpoints[0]
        assert endpoint.port == "https", f"Expected ServiceMonitor endpoint port 'https', got '{endpoint.port}'"
        assert endpoint.path == "/metrics", f"Expected ServiceMonitor endpoint path '/metrics', got '{endpoint.path}'"
        assert endpoint.interval == "30s", f"Expected ServiceMonitor scrape interval '30s', got '{endpoint.interval}'"
        assert endpoint.scheme == "https", f"Expected ServiceMonitor endpoint scheme 'https', got '{endpoint.scheme}'"

    def test_servicemonitor_owner_reference_to_evalhub(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_cr: EvalHub,
        evalhub_service_monitor: ServiceMonitor,
    ) -> None:
        """TC-SM-002: Verify ServiceMonitor carries ownerReference to EvalHub CR.

        The auto-created ServiceMonitor has an ownerReference pointing to the parent
        EvalHub CR, enabling Kubernetes garbage collection to automatically clean up
        the ServiceMonitor when the EvalHub CR is deleted.
        """
        owner_refs = evalhub_service_monitor.instance.metadata.ownerReferences
        assert owner_refs, (
            f"ServiceMonitor '{evalhub_service_monitor.name}' has no ownerReferences. "
            f"Expected ownerReference to EvalHub CR '{evalhub_cr.name}' for garbage collection."
        )

        evalhub_owner_ref = next(
            (ref for ref in owner_refs if ref.kind == "EvalHub"),
            None,
        )
        assert evalhub_owner_ref is not None, (
            f"ServiceMonitor '{evalhub_service_monitor.name}' has no ownerReference with kind 'EvalHub'. "
            f"Found ownerReferences: {[ref.kind for ref in owner_refs]}"
        )

        assert evalhub_owner_ref.name == evalhub_cr.name, (
            f"Expected ownerReference name '{evalhub_cr.name}', got '{evalhub_owner_ref.name}'"
        )
        assert evalhub_owner_ref.uid == evalhub_cr.instance.metadata.uid, (
            f"Expected ownerReference UID '{evalhub_cr.instance.metadata.uid}', got '{evalhub_owner_ref.uid}'"
        )
        assert evalhub_owner_ref.blockOwnerDeletion is True, (
            f"Expected ownerReference blockOwnerDeletion to be True for cascading delete, "
            f"got '{evalhub_owner_ref.blockOwnerDeletion}'"
        )

    def test_servicemonitor_auto_deleted_with_evalhub_cr(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_service_monitor: ServiceMonitor,
    ) -> None:
        """TC-SM-003: Verify ServiceMonitor auto-deleted via Kubernetes GC when EvalHub CR removed.

        Deleting an EvalHub CR triggers Kubernetes garbage collection to remove the
        associated ServiceMonitor resource due to the ownerReference relationship.
        This test creates and deletes a temporary EvalHub to verify GC behavior.
        """
        temp_evalhub_name = "evalhub-gc-test"
        with EvalHub(
            client=admin_client,
            name=temp_evalhub_name,
            namespace=model_namespace.name,
            database={"type": "sqlite"},
            wait_for_resource=True,
        ):
            temp_deployment = Deployment(
                client=admin_client,
                name=temp_evalhub_name,
                namespace=model_namespace.name,
            )
            temp_deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)

            temp_service_monitors = list(
                ServiceMonitor.get(
                    client=admin_client,
                    namespace=model_namespace.name,
                    label_selector=f"app={EVALHUB_APP_LABEL}",
                )
            )
            temp_sm = next(
                (sm for sm in temp_service_monitors if temp_evalhub_name in sm.name),
                None,
            )
            assert temp_sm is not None, f"No ServiceMonitor found for temporary EvalHub '{temp_evalhub_name}'"
            temp_sm_name = temp_sm.name

        # EvalHub CR deleted at context exit, wait for ServiceMonitor GC
        try:
            for _ in TimeoutSampler(
                wait_timeout=60,
                sleep=2,
                func=lambda: (
                    not ServiceMonitor(
                        client=admin_client,
                        name=temp_sm_name,
                        namespace=model_namespace.name,
                    ).exists
                ),
            ):
                if not ServiceMonitor(
                    client=admin_client,
                    name=temp_sm_name,
                    namespace=model_namespace.name,
                ).exists:
                    break
        except TimeoutExpiredError as err:
            msg = (
                f"ServiceMonitor '{temp_sm_name}' was not garbage collected within 60 seconds "
                f"after deleting EvalHub CR '{temp_evalhub_name}'. "
                f"Expected Kubernetes GC to remove ServiceMonitor due to ownerReference."
            )
            raise AssertionError(msg) from err


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-servicemonitor-update"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.model_explainability
class TestEvalHubServiceMonitorUpdate:
    """Tests for EvalHub ServiceMonitor update behavior (RHAISTRAT-1507)."""

    def test_servicemonitor_updated_on_config_change(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_cr: EvalHub,
        evalhub_service_monitor: ServiceMonitor,
    ) -> None:
        """TC-SM-004: Verify ServiceMonitor updated when EvalHub configuration changes.

        Modifying monitoring-relevant configuration in the EvalHub CR triggers the
        operator to update the associated ServiceMonitor resource. This is verified
        by checking the resourceVersion changes after updating the CR.
        """
        initial_resource_version = evalhub_service_monitor.instance.metadata.resourceVersion

        # Trigger a reconcile by patching the EvalHub CR (add a label)
        current_labels = evalhub_cr.instance.metadata.labels or {}
        current_labels["test-label"] = "trigger-reconcile"

        evalhub_cr.update(
            resource_dict={
                "metadata": {
                    "name": evalhub_cr.name,
                    "namespace": model_namespace.name,
                    "labels": current_labels,
                },
            }
        )

        # Wait for operator to reconcile and update ServiceMonitor
        try:
            for _ in TimeoutSampler(
                wait_timeout=60,
                sleep=2,
                func=lambda: (
                    ServiceMonitor(
                        client=admin_client,
                        name=evalhub_service_monitor.name,
                        namespace=model_namespace.name,
                    ).instance.metadata.resourceVersion
                    != initial_resource_version
                ),
            ):
                updated_sm = ServiceMonitor(
                    client=admin_client,
                    name=evalhub_service_monitor.name,
                    namespace=model_namespace.name,
                )
                if updated_sm.instance.metadata.resourceVersion != initial_resource_version:
                    # Verify ServiceMonitor still has correct configuration
                    endpoints = updated_sm.instance.spec.endpoints
                    assert endpoints, f"ServiceMonitor '{updated_sm.name}' has no endpoints after update"
                    endpoint = endpoints[0]
                    assert endpoint.port == "https", (
                        f"ServiceMonitor endpoint port changed after update. Expected 'https', got '{endpoint.port}'"
                    )
                    assert endpoint.path == "/metrics", (
                        f"ServiceMonitor endpoint path changed after update. Expected '/metrics', got '{endpoint.path}'"
                    )
                    assert endpoint.interval == "30s", (
                        f"ServiceMonitor interval changed after update. Expected '30s', got '{endpoint.interval}'"
                    )
                    break
        except TimeoutExpiredError as err:
            msg = (
                f"ServiceMonitor '{evalhub_service_monitor.name}' resourceVersion did not change within 60 seconds "
                f"after updating EvalHub CR. Initial version: {initial_resource_version}. "
                f"Expected operator to reconcile and update ServiceMonitor."
            )
            raise AssertionError(msg) from err


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-servicemonitor-disabled"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.model_explainability
class TestEvalHubServiceMonitorDisabled:
    """Tests for EvalHub ServiceMonitor when metrics disabled (RHAISTRAT-1507)."""

    def test_servicemonitor_not_created_when_metrics_disabled(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
    ) -> None:
        """TC-SM-005: Verify ServiceMonitor not created when prometheus.enabled is false.

        Deploying an EvalHub CR with prometheus.enabled: false should not create
        a ServiceMonitor resource. The EvalHub instance remains functional without
        Prometheus metrics scraping.
        """
        with EvalHub(
            client=admin_client,
            name="evalhub-no-metrics",
            namespace=model_namespace.name,
            database={"type": "sqlite"},
            prometheus={"enabled": False},
            wait_for_resource=True,
        ) as evalhub_no_metrics:
            deployment = Deployment(
                client=admin_client,
                name=evalhub_no_metrics.name,
                namespace=model_namespace.name,
            )
            deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)

            # Wait briefly to allow operator reconciliation, then verify no ServiceMonitor exists
            time.sleep(10)

            service_monitors = list(
                ServiceMonitor.get(
                    client=admin_client,
                    namespace=model_namespace.name,
                    label_selector=f"app={EVALHUB_APP_LABEL}",
                )
            )

            # Filter to only ServiceMonitors associated with this specific EvalHub instance
            evalhub_service_monitors = [sm for sm in service_monitors if evalhub_no_metrics.name in sm.name]

            assert not evalhub_service_monitors, (
                f"ServiceMonitor unexpectedly created for EvalHub '{evalhub_no_metrics.name}' "
                f"with prometheus.enabled=false. Found ServiceMonitors: {[sm.name for sm in evalhub_service_monitors]}"
            )

            # Verify EvalHub deployment is healthy despite no metrics
            assert deployment.instance.status.availableReplicas > 0, (
                f"EvalHub deployment '{deployment.name}' has no available replicas. "
                f"Expected EvalHub to be functional even with metrics disabled."
            )


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
@pytest.mark.model_explainability
class TestEvalHubServiceMonitorTLS:
    """Tests for EvalHub ServiceMonitor TLS configuration (RHAISTRAT-1507)."""

    def test_servicemonitor_tls_insecure_skip_verify(
        self,
        evalhub_service_monitor: ServiceMonitor,
    ) -> None:
        """TC-CFG-001: Verify ServiceMonitor TLS config has insecureSkipVerify enabled.

        The TrustyAI operator creates a ServiceMonitor with TLS configuration
        that includes insecureSkipVerify: true for compatibility with OpenShift's
        cluster-internal certificate authority. The endpoint scheme must be https.
        """
        endpoints = evalhub_service_monitor.instance.spec.endpoints
        assert endpoints, f"ServiceMonitor '{evalhub_service_monitor.name}' has no endpoints defined"

        endpoint = endpoints[0]

        assert endpoint.scheme == "https", f"Expected endpoint scheme 'https', got '{endpoint.scheme}'"

        tls_config = endpoint.tlsConfig
        assert tls_config is not None, f"ServiceMonitor '{evalhub_service_monitor.name}' endpoint has no tlsConfig"

        assert tls_config.insecureSkipVerify is True, (
            f"Expected tlsConfig.insecureSkipVerify to be true, got '{tls_config.insecureSkipVerify}'"
        )


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-servicemonitor-port-path"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier2
@pytest.mark.model_explainability
class TestEvalHubServiceMonitorConfiguration:
    """Tests for EvalHub ServiceMonitor port and path configuration (RHAISTRAT-1507)."""

    def test_servicemonitor_targets_correct_port_and_path(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_service_monitor: ServiceMonitor,
    ) -> None:
        """TC-CFG-002: Verify ServiceMonitor targets correct port and path.

        The ServiceMonitor targets the EvalHub Service on the port named 'https'
        at path '/metrics', ensuring Prometheus scrapes the correct endpoint.
        """
        endpoints = evalhub_service_monitor.instance.spec.endpoints
        assert endpoints, f"ServiceMonitor '{evalhub_service_monitor.name}' has no endpoints defined"

        endpoint = endpoints[0]

        assert endpoint.port == "https", f"Expected ServiceMonitor endpoint port name 'https', got '{endpoint.port}'"
        assert endpoint.path == "/metrics", f"Expected ServiceMonitor endpoint path '/metrics', got '{endpoint.path}'"

        # Verify the EvalHub Service has a port named 'https'
        evalhub_services = list(
            Service.get(
                client=admin_client,
                namespace=model_namespace.name,
                label_selector=f"app={EVALHUB_APP_LABEL}",
            )
        )

        assert evalhub_services, (
            f"No EvalHub Service found in namespace '{model_namespace.name}' with label app={EVALHUB_APP_LABEL}"
        )

        evalhub_service = evalhub_services[0]
        service_ports = evalhub_service.instance.spec.ports

        https_port = next(
            (port for port in service_ports if port.name == "https"),
            None,
        )

        assert https_port is not None, (
            f"EvalHub Service '{evalhub_service.name}' does not expose a port named 'https'. "
            f"Found ports: {[port.name for port in service_ports]}. "
            f"ServiceMonitor targets port 'https' so the Service must expose it."
        )
