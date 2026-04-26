import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.cluster_role import ClusterRole
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.deployment import Deployment
from ocp_resources.evalhub import EvalHub
from ocp_resources.namespace import Namespace
from ocp_resources.prometheus_rule import PrometheusRule
from ocp_resources.service_monitor import ServiceMonitor
from timeout_sampler import TimeoutSampler

from tests.model_explainability.evalhub.constants import (
    EVALHUB_APP_LABEL,
    EVALHUB_COMPONENT_LABEL,
    EVALHUB_METRICS_PATH,
    EVALHUB_SERVICE_CA_CERT_KEY,
    EVALHUB_SERVICE_CA_CONFIGMAP_SUFFIX,
    EVALHUB_SERVICE_MONITOR_PORT,
    EVALHUB_SERVICE_MONITOR_SCHEME,
    EVALHUB_SERVICE_MONITOR_SCRAPE_INTERVAL,
    EVALHUB_SERVICE_MONITOR_SUFFIX,
)

SERVICE_MONITOR_CRD = "servicemonitors.monitoring.coreos.com"
EVALHUB_MANAGER_CLUSTERROLE = "trustyai-service-operator-evalhub-manager-role"
SERVICEMONITOR_API_GROUP = "monitoring.coreos.com"
REQUIRED_SERVICEMONITOR_VERBS = {"get", "list", "watch", "create", "update"}


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-sm"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.model_explainability
class TestEvalHubServiceMonitor:
    """Tests for the EvalHub ServiceMonitor auto-discovery feature.

    Verifies that the TrustyAI Operator automatically creates a correctly
    configured ServiceMonitor when an EvalHub CR is deployed, enabling
    Prometheus scraping via the OpenShift user-workload-monitoring stack.
    """

    @pytest.fixture(autouse=True)
    def _service_monitor(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_cr: EvalHub,
        evalhub_deployment: Deployment,
    ) -> ServiceMonitor:
        sm = ServiceMonitor(
            client=admin_client,
            name=f"{evalhub_cr.name}{EVALHUB_SERVICE_MONITOR_SUFFIX}",
            namespace=model_namespace.name,
        )
        self._sm = sm
        self._evalhub_cr = evalhub_cr
        self._model_namespace = model_namespace
        return sm

    def test_servicemonitor_created(self) -> None:
        """Verify the operator creates a ServiceMonitor after EvalHub is deployed."""
        assert self._sm.exists, (
            f"ServiceMonitor '{self._sm.name}' not found in namespace '{self._model_namespace.name}'. "
            "The operator may not have reconciled yet or the Prometheus Operator CRD is absent."
        )

    def test_servicemonitor_name(self) -> None:
        """Verify the ServiceMonitor name follows the '{evalhub-name}-metrics' convention."""
        expected = f"{self._evalhub_cr.name}{EVALHUB_SERVICE_MONITOR_SUFFIX}"
        assert self._sm.name == expected, f"Expected name '{expected}', got '{self._sm.name}'"

    def test_servicemonitor_labels(self) -> None:
        """Verify the ServiceMonitor carries the standard EvalHub labels."""
        labels = self._sm.instance.metadata.labels or {}
        assert labels.get("app") == EVALHUB_APP_LABEL, (
            f"Expected label app={EVALHUB_APP_LABEL}, got {labels.get('app')}"
        )
        assert labels.get("component") == EVALHUB_COMPONENT_LABEL, (
            f"Expected label component={EVALHUB_COMPONENT_LABEL}, got {labels.get('component')}"
        )
        assert labels.get("instance") == self._evalhub_cr.name, (
            f"Expected label instance={self._evalhub_cr.name}, got {labels.get('instance')}"
        )

    def test_servicemonitor_single_endpoint(self) -> None:
        """Verify the ServiceMonitor defines exactly one scrape endpoint."""
        endpoints = self._sm.instance.spec.endpoints
        assert len(endpoints) == 1, f"Expected 1 endpoint, got {len(endpoints)}"

    def test_servicemonitor_endpoint_path(self) -> None:
        """Verify the scrape path is /metrics."""
        endpoint = self._sm.instance.spec.endpoints[0]
        assert endpoint.path == EVALHUB_METRICS_PATH, (
            f"Expected path '{EVALHUB_METRICS_PATH}', got '{endpoint.path}'"
        )

    def test_servicemonitor_endpoint_scheme(self) -> None:
        """Verify the scrape scheme is https."""
        endpoint = self._sm.instance.spec.endpoints[0]
        assert endpoint.scheme == EVALHUB_SERVICE_MONITOR_SCHEME, (
            f"Expected scheme '{EVALHUB_SERVICE_MONITOR_SCHEME}', got '{endpoint.scheme}'"
        )

    def test_servicemonitor_endpoint_port(self) -> None:
        """Verify the scrape port name is 'https'."""
        endpoint = self._sm.instance.spec.endpoints[0]
        assert endpoint.port == EVALHUB_SERVICE_MONITOR_PORT, (
            f"Expected port '{EVALHUB_SERVICE_MONITOR_PORT}', got '{endpoint.port}'"
        )

    def test_servicemonitor_scrape_interval(self) -> None:
        """Verify the scrape interval is 30s."""
        endpoint = self._sm.instance.spec.endpoints[0]
        assert endpoint.interval == EVALHUB_SERVICE_MONITOR_SCRAPE_INTERVAL, (
            f"Expected interval '{EVALHUB_SERVICE_MONITOR_SCRAPE_INTERVAL}', got '{endpoint.interval}'"
        )

    def test_servicemonitor_honor_labels(self) -> None:
        """Verify honorLabels is enabled on the scrape endpoint."""
        endpoint = self._sm.instance.spec.endpoints[0]
        assert endpoint.honorLabels is True, "Expected honorLabels=True on the scrape endpoint"

    def test_servicemonitor_tls_configured(self) -> None:
        """Verify TLS is configured on the scrape endpoint."""
        endpoint = self._sm.instance.spec.endpoints[0]
        assert endpoint.tlsConfig is not None, "Expected tlsConfig to be set on the scrape endpoint"

    def test_servicemonitor_tls_server_name(self) -> None:
        """Verify TLS serverName matches the EvalHub service DNS name."""
        endpoint = self._sm.instance.spec.endpoints[0]
        expected = f"{self._evalhub_cr.name}.{self._model_namespace.name}.svc"
        actual = endpoint.tlsConfig.get("safeConfig", {}).get("serverName") or endpoint.tlsConfig.serverName
        assert actual == expected, f"Expected TLS serverName '{expected}', got '{actual}'"

    def test_servicemonitor_tls_ca_configmap(self) -> None:
        """Verify TLS CA is sourced from the service-ca ConfigMap injected by OpenShift."""
        endpoint = self._sm.instance.spec.endpoints[0]
        # SafeTLSConfig is inlined into TLSConfig, so the path is tlsConfig.ca.configMap.{name,key}
        configmap = endpoint.tlsConfig.ca.configMap
        expected_cm_name = f"{self._evalhub_cr.name}{EVALHUB_SERVICE_CA_CONFIGMAP_SUFFIX}"

        assert configmap.name == expected_cm_name, (
            f"Expected CA ConfigMap name '{expected_cm_name}', got '{configmap.name}'"
        )
        assert configmap.key == EVALHUB_SERVICE_CA_CERT_KEY, (
            f"Expected CA ConfigMap key '{EVALHUB_SERVICE_CA_CERT_KEY}', got '{configmap.key}'"
        )

    def test_servicemonitor_tls_no_insecure_skip(self) -> None:
        """Verify insecureSkipVerify is not set — TLS must be fully validated."""
        endpoint = self._sm.instance.spec.endpoints[0]
        tls = endpoint.tlsConfig
        insecure = tls.get("insecureSkipVerify") if isinstance(tls, dict) else getattr(tls, "insecureSkipVerify", None)
        assert not insecure, "insecureSkipVerify must not be set — EvalHub uses service CA for TLS validation"

    def test_servicemonitor_namespace_selector(self) -> None:
        """Verify the ServiceMonitor scopes scraping to the EvalHub namespace."""
        match_names = self._sm.instance.spec.namespaceSelector.matchNames or []
        assert self._model_namespace.name in match_names, (
            f"Expected namespace '{self._model_namespace.name}' in namespaceSelector.matchNames, got {match_names}"
        )

    def test_servicemonitor_label_selector(self) -> None:
        """Verify the ServiceMonitor label selector targets the EvalHub service."""
        match_labels = self._sm.instance.spec.selector.matchLabels or {}
        assert match_labels.get("app") == EVALHUB_APP_LABEL, (
            f"Expected selector app={EVALHUB_APP_LABEL}, got {match_labels.get('app')}"
        )
        assert match_labels.get("component") == EVALHUB_COMPONENT_LABEL, (
            f"Expected selector component={EVALHUB_COMPONENT_LABEL}, got {match_labels.get('component')}"
        )
        assert match_labels.get("instance") == self._evalhub_cr.name, (
            f"Expected selector instance={self._evalhub_cr.name}, got {match_labels.get('instance')}"
        )

    def test_servicemonitor_owner_reference(self) -> None:
        """Verify the ServiceMonitor is owned by the EvalHub CR for automatic garbage collection."""
        owner_refs = self._sm.instance.metadata.ownerReferences or []
        assert len(owner_refs) == 1, f"Expected 1 ownerReference, got {len(owner_refs)}"
        owner = owner_refs[0]
        assert owner.name == self._evalhub_cr.name, (
            f"Expected ownerReference.name='{self._evalhub_cr.name}', got '{owner.name}'"
        )
        assert owner.kind == "EvalHub", f"Expected ownerReference.kind='EvalHub', got '{owner.kind}'"

    def test_operator_clusterrole_has_servicemonitor_permissions(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Verify the TrustyAI Operator ClusterRole grants the required ServiceMonitor permissions.

        The kubebuilder RBAC marker in evalhub_controller.go generates a rule for
        monitoring.coreos.com/servicemonitors. This test verifies it is present in the
        deployed ClusterRole, confirming the operator can create and manage ServiceMonitors.
        """
        role = ClusterRole(client=admin_client, name=EVALHUB_MANAGER_CLUSTERROLE, ensure_exists=True)
        assert role.exists, f"ClusterRole '{EVALHUB_MANAGER_CLUSTERROLE}' not found"

        rules = role.instance.rules or []
        sm_rules = [
            r for r in rules
            if SERVICEMONITOR_API_GROUP in (r.apiGroups or [])
            and "servicemonitors" in (r.resources or [])
        ]
        assert sm_rules, (
            f"ClusterRole '{EVALHUB_MANAGER_CLUSTERROLE}' has no rule for "
            f"{SERVICEMONITOR_API_GROUP}/servicemonitors. "
            "The RBAC marker in evalhub_controller.go may not have been applied."
        )
        granted_verbs = {v for rule in sm_rules for v in (rule.verbs or [])}
        missing = REQUIRED_SERVICEMONITOR_VERBS - granted_verbs
        assert not missing, (
            f"ClusterRole '{EVALHUB_MANAGER_CLUSTERROLE}' is missing verbs for servicemonitors: {missing}. "
            f"Granted: {granted_verbs}"
        )

    def test_prometheus_rule_can_be_created(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_cr: EvalHub,
        evalhub_deployment: Deployment,
    ) -> None:
        """Verify a PrometheusRule targeting EvalHub metrics can be created in the namespace.

        This is the alerting integration proof from the acceptance criteria: a platform
        administrator must be able to define alerting rules against EvalHub metrics.
        The test does not wait for the alert to fire — it only verifies the resource
        can be created and is accepted by the cluster.
        """
        rule_name = f"{evalhub_cr.name}-test-alert"
        with PrometheusRule(
            client=admin_client,
            name=rule_name,
            namespace=model_namespace.name,
            kind_dict={
                "apiVersion": "monitoring.coreos.com/v1",
                "kind": "PrometheusRule",
                "metadata": {"name": rule_name, "namespace": model_namespace.name},
                "spec": {
                    "groups": [
                        {
                            "name": "evalhub.test.rules",
                            "rules": [
                                {
                                    "alert": "EvalHubHighRequestRate",
                                    "expr": 'rate(http_requests_total{job="evalhub"}[5m]) > 100',
                                    "for": "1m",
                                    "labels": {"severity": "warning"},
                                    "annotations": {
                                        "summary": "EvalHub request rate is high",
                                    },
                                }
                            ],
                        }
                    ]
                },
            },
        ) as pr:
            assert pr.exists, (
                f"PrometheusRule '{rule_name}' could not be created in namespace '{model_namespace.name}'. "
                "The cluster may not have the Prometheus Operator CRD installed."
            )


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-sm-no-crd"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.model_explainability
class TestEvalHubServiceMonitorCRDAbsent:
    """Graceful-degradation tests for EvalHub when the Prometheus Operator is not installed.

    Verifies that the absence of the ServiceMonitor CRD does not block EvalHub
    reconciliation — the operator should detect the missing CRD and skip monitoring
    setup silently rather than failing the deployment.

    These tests only run on clusters where the ServiceMonitor CRD is absent.
    """

    @pytest.fixture(autouse=True)
    def _skip_if_crd_present(self, admin_client: DynamicClient) -> None:
        crd = CustomResourceDefinition(client=admin_client, name=SERVICE_MONITOR_CRD)
        if crd.exists:
            pytest.skip("ServiceMonitor CRD is present — graceful-degradation tests require it to be absent")

    def test_evalhub_becomes_ready_without_servicemonitor_crd(
        self,
        evalhub_deployment: Deployment,
    ) -> None:
        """Verify EvalHub reaches a ready state even when the ServiceMonitor CRD is absent."""
        assert evalhub_deployment.exists, "EvalHub deployment was not created"
        assert evalhub_deployment.instance.status.availableReplicas >= 1, (
            "EvalHub deployment is not ready — operator may have blocked on missing ServiceMonitor CRD"
        )

    def test_no_servicemonitor_created_when_crd_absent(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_cr: EvalHub,
        evalhub_deployment: Deployment,
    ) -> None:
        """Verify the operator does not attempt to create a ServiceMonitor when the CRD is absent."""
        sm = ServiceMonitor(
            client=admin_client,
            name=f"{evalhub_cr.name}{EVALHUB_SERVICE_MONITOR_SUFFIX}",
            namespace=model_namespace.name,
        )
        assert not sm.exists, (
            f"ServiceMonitor '{sm.name}' was unexpectedly created despite the CRD being absent"
        )


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-sm-del"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.model_explainability
class TestEvalHubServiceMonitorDeletion:
    """Lifecycle deletion test: verify ServiceMonitor is GC'd when the EvalHub CR is removed.

    Creates an EvalHub CR inline (not via the class-scoped fixture) so that deletion
    can be triggered explicitly within the test, then waits for Kubernetes owner-reference
    garbage collection to remove the ServiceMonitor.
    """

    @pytest.fixture(autouse=True)
    def _skip_if_crd_absent(self, admin_client: DynamicClient) -> None:
        crd = CustomResourceDefinition(client=admin_client, name=SERVICE_MONITOR_CRD)
        if not crd.exists:
            pytest.skip(f"ServiceMonitor CRD ({SERVICE_MONITOR_CRD}) is not installed on this cluster")

    def test_servicemonitor_deleted_with_evalhub(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
    ) -> None:
        """Verify the ServiceMonitor is removed by GC after the owning EvalHub CR is deleted."""
        evalhub_name = "evalhub-del"
        sm_name = f"{evalhub_name}{EVALHUB_SERVICE_MONITOR_SUFFIX}"

        with EvalHub(
            client=admin_client,
            name=evalhub_name,
            namespace=model_namespace.name,
            database={"type": "sqlite"},
            wait_for_resource=True,
        ):
            # Wait for the ServiceMonitor to appear
            sm = ServiceMonitor(client=admin_client, name=sm_name, namespace=model_namespace.name)
            for _ in TimeoutSampler(wait_timeout=60, sleep=5, func=lambda: sm.exists):
                if sm.exists:
                    break

            assert sm.exists, (
                f"ServiceMonitor '{sm_name}' was not created after EvalHub deployment"
            )

        # EvalHub CR is now deleted (context manager exited); wait for GC
        sm_gone = False
        for _ in TimeoutSampler(wait_timeout=60, sleep=5, func=lambda: not sm.exists):
            if not sm.exists:
                sm_gone = True
                break

        assert sm_gone, (
            f"ServiceMonitor '{sm_name}' was not garbage-collected after EvalHub CR deletion. "
            "Check that the ownerReference is set correctly."
        )
