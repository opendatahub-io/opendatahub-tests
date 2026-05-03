import pytest
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.evalhub import EvalHub
from ocp_resources.namespace import Namespace
from ocp_resources.network_policy import NetworkPolicy
from pytest_testconfig import config as py_config

from tests.model_explainability.evalhub.constants import (
    EVALHUB_APP_LABEL,
    EVALHUB_COMPONENT_LABEL,
    EVALHUB_SERVICE_PORT,
)

CLUSTER_MONITORING_CONFIGMAP = "cluster-monitoring-config"
OPENSHIFT_MONITORING_NAMESPACE = "openshift-monitoring"

# Namespace where OpenShift user-workload-monitoring Prometheus runs
MONITORING_NAMESPACE: str = "openshift-user-workload-monitoring"

# Label OpenShift sets on every namespace — used in NetworkPolicy namespaceSelectors
NAMESPACE_NAME_LABEL: str = "kubernetes.io/metadata.name"


@pytest.mark.smoke
@pytest.mark.model_explainability
def test_user_workload_monitoring_enabled(admin_client: DynamicClient) -> None:
    """Verify user-workload-monitoring is enabled on this cluster.

    EvalHub ServiceMonitor integration requires user-workload-monitoring to be enabled
    in the cluster-monitoring-config ConfigMap. Without it, Prometheus will not pick up
    ServiceMonitors in user namespaces.
    """
    cm = ConfigMap(
        client=admin_client,
        name=CLUSTER_MONITORING_CONFIGMAP,
        namespace=OPENSHIFT_MONITORING_NAMESPACE,
    )
    assert cm.exists, (
        f"ConfigMap '{CLUSTER_MONITORING_CONFIGMAP}' not found in '{OPENSHIFT_MONITORING_NAMESPACE}'. "
        "User-workload-monitoring may not be configured on this cluster."
    )
    config_yaml = (cm.instance.data or {}).get("config.yaml", "")
    config = yaml.safe_load(config_yaml) or {}
    assert config.get("enableUserWorkload") is True, (
        f"'enableUserWorkload: true' is not set in '{CLUSTER_MONITORING_CONFIGMAP}'. "
        "EvalHub metrics will not be scraped by user-workload-monitoring Prometheus."
    )


@pytest.mark.smoke
@pytest.mark.model_explainability
def test_monitoring_namespace_carries_name_label(admin_client: DynamicClient) -> None:
    """Verify the monitoring namespace has the kubernetes.io/metadata.name label.

    NetworkPolicy namespaceSelectors use this label to identify the monitoring
    namespace. If the label is absent, allow rules targeting the namespace by name
    will silently fail to match.
    """
    ns = Namespace(client=admin_client, name=MONITORING_NAMESPACE, ensure_exists=True)
    assert ns.exists, f"Namespace '{MONITORING_NAMESPACE}' does not exist on this cluster"
    labels = ns.instance.metadata.labels or {}
    assert labels.get(NAMESPACE_NAME_LABEL) == MONITORING_NAMESPACE, (
        f"Expected namespace '{MONITORING_NAMESPACE}' to carry label "
        f"'{NAMESPACE_NAME_LABEL}={MONITORING_NAMESPACE}', got: {labels.get(NAMESPACE_NAME_LABEL)}"
    )


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-netpol"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.model_explainability
class TestEvalHubNetworkPolicy:
    """Tests verifying that NetworkPolicy configuration allows Prometheus to scrape EvalHub metrics.

    PR #713 removed the operator-managed NetworkPolicy in favour of external network
    management. These tests verify that:
    - The operator no longer creates a NetworkPolicy for EvalHub.
    - No NetworkPolicy in the EvalHub namespace blocks ingress on the metrics port
      from the openshift-user-workload-monitoring namespace.
    - The monitoring namespace carries the expected label used by NetworkPolicy selectors.
    """

    def test_operator_does_not_create_networkpolicy(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_cr: EvalHub,
        evalhub_deployment: Deployment,
    ) -> None:
        """Verify the operator no longer creates a NetworkPolicy for EvalHub metrics access.

        The operator-managed NetworkPolicy was removed in trustyai-service-operator#713
        in favour of external network management. No evalhub-named NetworkPolicy should
        exist in the namespace.
        """
        policies = list(NetworkPolicy.get(client=admin_client, namespace=model_namespace.name))
        cr_uid = evalhub_cr.instance.metadata.uid

        def _owned_by_evalhub_cr(policy: NetworkPolicy) -> bool:
            for ref in policy.instance.metadata.get("ownerReferences") or []:
                if ref.get("kind") == "EvalHub" and ref.get("name") == evalhub_cr.name:
                    if cr_uid is None or ref.get("uid") == cr_uid:
                        return True
            return False

        evalhub_netpols = [p for p in policies if _owned_by_evalhub_cr(p)]
        assert not evalhub_netpols, (
            f"Operator unexpectedly created NetworkPolicy resources for EvalHub: "
            f"{[p.name for p in evalhub_netpols]}. "
            "These should have been removed in trustyai-service-operator#713."
        )

    def test_no_networkpolicy_blocks_metrics_port_from_monitoring(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_cr: EvalHub,
        evalhub_deployment: Deployment,
    ) -> None:
        """Verify no NetworkPolicy in the namespace denies ingress on the metrics port from the monitoring namespace.

        For each NetworkPolicy that selects EvalHub pods and restricts port 8443, the
        monitoring namespace must appear in the allowed ingress sources.
        """
        evalhub_labels = {
            "app": EVALHUB_APP_LABEL,
            "component": EVALHUB_COMPONENT_LABEL,
        }

        policies = list(NetworkPolicy.get(client=admin_client, namespace=model_namespace.name))
        for policy in policies:
            spec = policy.instance.spec

            # Only care about policies whose podSelector targets EvalHub pods
            match_labels = (spec.podSelector or {}).get("matchLabels", {})
            if not all(match_labels.get(k) == v for k, v in evalhub_labels.items()):
                continue

            # Only care about policies that restrict the metrics port
            ingress_rules = spec.get("ingress") or []
            for rule in ingress_rules:
                ports = rule.get("ports") or []
                port_numbers = {p.get("port") for p in ports}
                if EVALHUB_SERVICE_PORT not in port_numbers:
                    continue

                # A rule restricts port 8443 — verify monitoring namespace is in the allow list
                from_rules = rule.get("from") or []
                monitoring_allowed = any(
                    (ns_sel := from_rule.get("namespaceSelector", {}))
                    and ns_sel.get("matchLabels", {}).get(NAMESPACE_NAME_LABEL) == MONITORING_NAMESPACE
                    for from_rule in from_rules
                )
                assert monitoring_allowed, (
                    f"NetworkPolicy '{policy.name}' restricts port {EVALHUB_SERVICE_PORT} "
                    f"on EvalHub pods but does not allow ingress from '{MONITORING_NAMESPACE}'. "
                    "Prometheus will not be able to scrape EvalHub metrics."
                )

    def test_applications_namespace_evalhub_servicemonitor_reachable(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_cr: EvalHub,
        evalhub_deployment: Deployment,
    ) -> None:
        """Verify no NetworkPolicy in the applications namespace blocks the metrics port.

        EvalHub is typically deployed in the applications namespace in production
        (e.g. redhat-ods-applications). This test checks the same port-blocking logic
        against NetworkPolicies in that namespace, which may have stricter default-deny rules.
        """
        applications_namespace = py_config["applications_namespace"]
        if applications_namespace == model_namespace.name:
            pytest.skip("EvalHub test namespace is the applications namespace — already covered above")

        evalhub_labels = {
            "app": EVALHUB_APP_LABEL,
            "component": EVALHUB_COMPONENT_LABEL,
        }

        policies = list(NetworkPolicy.get(client=admin_client, namespace=applications_namespace))
        for policy in policies:
            spec = policy.instance.spec

            # Only consider policies whose podSelector targets EvalHub pods
            match_labels = (spec.podSelector or {}).get("matchLabels", {})
            if not all(match_labels.get(k) == v for k, v in evalhub_labels.items()):
                continue

            ingress_rules = spec.get("ingress") or []
            for rule in ingress_rules:
                ports = rule.get("ports") or []
                port_numbers = {p.get("port") for p in ports}
                if EVALHUB_SERVICE_PORT not in port_numbers:
                    continue

                from_rules = rule.get("from") or []
                monitoring_allowed = any(
                    (ns_sel := from_rule.get("namespaceSelector", {}))
                    and ns_sel.get("matchLabels", {}).get(NAMESPACE_NAME_LABEL) == MONITORING_NAMESPACE
                    for from_rule in from_rules
                )
                assert monitoring_allowed or not from_rules, (
                    f"NetworkPolicy '{policy.name}' in '{applications_namespace}' restricts "
                    f"port {EVALHUB_SERVICE_PORT} on EvalHub pods without allowing '{MONITORING_NAMESPACE}'. "
                    "This would block Prometheus from scraping EvalHub metrics in production."
                )
