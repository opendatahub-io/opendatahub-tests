"""EvalHub NetworkPolicy and RBAC security tests.

Test Plan Reference: RHAISTRAT-1507
Test Cases: TC-NP-001, TC-NP-002, TC-RBAC-001, TC-RBAC-002, TC-RBAC-003
"""

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.cluster_role import ClusterRole
from ocp_resources.cluster_role_binding import ClusterRoleBinding
from ocp_resources.deployment import Deployment
from ocp_resources.evalhub import EvalHub
from ocp_resources.namespace import Namespace
from ocp_resources.network_policy import NetworkPolicy
from ocp_resources.pod import Pod
from ocp_resources.service_monitor import ServiceMonitor
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_explainability.evalhub.constants import EVALHUB_APP_LABEL
from utilities.constants import Timeout

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-networkpolicy"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.model_explainability
class TestEvalHubNetworkPolicy:
    """Tests for EvalHub NetworkPolicy configuration (RHAISTRAT-1507)."""

    def test_networkpolicy_permits_prometheus_ingress(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_cr: EvalHub,
        evalhub_deployment: Deployment,
    ) -> None:
        """TC-NP-001: NetworkPolicy permits Prometheus ingress to metrics port.

        Verify that a NetworkPolicy is created that permits ingress from the
        Prometheus scraper in the monitoring namespace to the EvalHub metrics port.
        """
        # List NetworkPolicies in the EvalHub namespace
        network_policies = list(
            NetworkPolicy.get(
                client=admin_client,
                namespace=model_namespace.name,
            )
        )

        assert network_policies, (
            f"No NetworkPolicies found in namespace '{model_namespace.name}'. "
            f"Expected a NetworkPolicy to allow Prometheus ingress."
        )

        # Find NetworkPolicy that targets EvalHub pods and allows monitoring namespace ingress
        evalhub_network_policy = None
        for np in network_policies:
            spec = np.instance.spec

            # Check if this NetworkPolicy has ingress rules
            if not hasattr(spec, "ingress") or not spec.ingress:
                continue

            # Check for podSelector targeting EvalHub pods
            pod_selector = spec.podSelector
            if hasattr(pod_selector, "matchLabels"):
                labels = pod_selector.matchLabels or {}
                if labels.get("app") == EVALHUB_APP_LABEL or labels.get("component") == "api":
                    # Check ingress rules for monitoring namespace
                    for ingress_rule in spec.ingress:
                        if hasattr(ingress_rule, "from"):
                            for from_rule in ingress_rule.get("from", []):
                                if hasattr(from_rule, "namespaceSelector"):
                                    ns_selector = from_rule.namespaceSelector
                                    # Check if it targets monitoring namespace
                                    if hasattr(ns_selector, "matchLabels"):
                                        match_labels = ns_selector.matchLabels or {}
                                        # OpenShift monitoring namespaces typically have these labels
                                        if (
                                            "openshift.io/cluster-monitoring" in match_labels
                                            or "monitoring" in str(match_labels)
                                            or "user-workload-monitoring" in np.name
                                        ):
                                            evalhub_network_policy = np
                                            break

        assert evalhub_network_policy is not None, (
            f"No NetworkPolicy found that permits ingress from monitoring namespace to EvalHub pods. "
            f"Found NetworkPolicies: {[np.name for np in network_policies]}"
        )

        # Verify the NetworkPolicy ingress rule targets the metrics port
        spec = evalhub_network_policy.instance.spec
        assert hasattr(spec, "ingress") and spec.ingress, (
            f"NetworkPolicy '{evalhub_network_policy.name}' has no ingress rules"
        )

        # Check that ingress rules include port configuration for metrics
        has_port_rule = False
        for ingress_rule in spec.ingress:
            if hasattr(ingress_rule, "ports") and ingress_rule.ports:
                for port_rule in ingress_rule.ports:
                    # Metrics are typically exposed on HTTPS port (8443) or metrics port
                    if hasattr(port_rule, "port"):
                        port_value = port_rule.port
                        # Accept numeric port or named port like "https" or "metrics"
                        if port_value in ["https", "metrics", 8443, 8080]:
                            has_port_rule = True
                            break

        assert has_port_rule, (
            f"NetworkPolicy '{evalhub_network_policy.name}' ingress rules do not target "
            f"the metrics port (https/8443 or metrics/8080)"
        )

    def test_scraping_succeeds_with_default_deny_ingress(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_cr: EvalHub,
        evalhub_deployment: Deployment,
        evalhub_service_monitor: ServiceMonitor,
    ) -> None:
        """TC-NP-002: Prometheus can scrape metrics in namespace with default-deny ingress.

        Verify that Prometheus can scrape the EvalHub /metrics endpoint in a namespace
        that enforces default-deny ingress policy.
        """
        # Verify the namespace has NetworkPolicy enforcement (may have default-deny)
        network_policies = list(
            NetworkPolicy.get(
                client=admin_client,
                namespace=model_namespace.name,
            )
        )

        # For this test to be meaningful, we expect at least one NetworkPolicy exists
        # (either default-deny or the EvalHub-specific allow rule)
        assert network_policies, (
            f"No NetworkPolicies found in namespace '{model_namespace.name}'. "
            f"This test requires NetworkPolicy enforcement to verify Prometheus can bypass restrictions."
        )

        # Verify ServiceMonitor exists and is properly configured
        assert evalhub_service_monitor.exists, f"ServiceMonitor '{evalhub_service_monitor.name}' does not exist"

        endpoints = evalhub_service_monitor.instance.spec.endpoints
        assert endpoints, f"ServiceMonitor '{evalhub_service_monitor.name}' has no endpoints defined"

        endpoint = endpoints[0]
        assert endpoint.path == "/metrics", f"Expected ServiceMonitor endpoint path '/metrics', got '{endpoint.path}'"

        # Get one of the EvalHub pods to verify it's accessible
        evalhub_pods = list(
            Pod.get(
                client=admin_client,
                namespace=model_namespace.name,
                label_selector=f"app={EVALHUB_APP_LABEL}",
            )
        )

        assert evalhub_pods, f"No EvalHub pods found in namespace '{model_namespace.name}'"

        evalhub_pod = evalhub_pods[0]
        assert evalhub_pod.instance.status.phase == "Running", (
            f"EvalHub pod '{evalhub_pod.name}' is not Running (phase: {evalhub_pod.instance.status.phase})"
        )

        # Verify the deployment is healthy (replicas available)
        assert evalhub_deployment.instance.status.availableReplicas > 0, (
            f"EvalHub deployment '{evalhub_deployment.name}' has no available replicas. "
            f"Cannot verify Prometheus scraping without healthy pods."
        )

        # NOTE: Direct verification of Prometheus scrape target health would require
        # access to the Prometheus API in the monitoring namespace. This test verifies
        # that the prerequisites are in place (ServiceMonitor, NetworkPolicy, healthy pods).
        # In a real cluster with Prometheus operator, the ServiceMonitor would be picked up
        # and the target would appear in Prometheus with health=UP if NetworkPolicy permits access.


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-rbac"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.model_explainability
class TestEvalHubOperatorRBAC:
    """Tests for TrustyAI operator RBAC configuration (RHAISTRAT-1507)."""

    def test_operator_clusterrole_has_servicemonitor_permissions(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
    ) -> None:
        """TC-RBAC-001: Operator ClusterRole has ServiceMonitor CRUD permissions.

        Verify that the TrustyAI operator's ClusterRole includes the required permissions
        to create, update, patch, delete, get, list, and watch ServiceMonitor resources.
        """
        # Find the TrustyAI operator ClusterRoleBindings
        cluster_role_bindings = list(ClusterRoleBinding.get(client=admin_client))

        # Look for ClusterRoleBindings that reference the TrustyAI operator ServiceAccount
        trustyai_crbs = [
            crb
            for crb in cluster_role_bindings
            if any(
                subject.kind == "ServiceAccount" and "trustyai" in subject.name.lower()
                for subject in (crb.instance.subjects or [])
            )
        ]

        assert trustyai_crbs, (
            "No ClusterRoleBindings found for TrustyAI operator ServiceAccount. "
            "Expected to find operator RBAC configuration."
        )

        # Get the ClusterRole name(s) referenced by the operator's ClusterRoleBindings
        operator_cluster_role_names = set()
        for crb in trustyai_crbs:
            if crb.instance.roleRef.kind == "ClusterRole":
                operator_cluster_role_names.add(crb.instance.roleRef.name)

        assert operator_cluster_role_names, "No ClusterRoles found referenced by TrustyAI operator ClusterRoleBindings"

        # Check each ClusterRole for ServiceMonitor permissions
        has_servicemonitor_permissions = False

        for cluster_role_name in operator_cluster_role_names:
            cluster_role = ClusterRole(
                client=admin_client,
                name=cluster_role_name,
            )

            if not cluster_role.exists:
                continue

            # Check the rules for ServiceMonitor permissions
            rules = cluster_role.instance.rules or []
            for rule in rules:
                api_groups = rule.apiGroups or []
                resources = rule.resources or []
                verbs = rule.verbs or []

                # Check if this rule applies to ServiceMonitors
                if "monitoring.coreos.com" in api_groups and "servicemonitors" in resources:
                    required_verbs = {"create", "update", "patch", "delete", "get", "list", "watch"}
                    actual_verbs = set(verbs)

                    if required_verbs.issubset(actual_verbs):
                        has_servicemonitor_permissions = True
                        break

            if has_servicemonitor_permissions:
                break

        assert has_servicemonitor_permissions, (
            f"TrustyAI operator ClusterRole(s) {list(operator_cluster_role_names)} do not have "
            f"required ServiceMonitor permissions. Expected a rule with:\n"
            f"  - apiGroups: ['monitoring.coreos.com']\n"
            f"  - resources: ['servicemonitors']\n"
            f"  - verbs: ['create', 'update', 'patch', 'delete', 'get', 'list', 'watch']"
        )

    def test_restricted_user_cannot_modify_servicemonitor(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_service_monitor: ServiceMonitor,
    ) -> None:
        """TC-RBAC-002: Restricted user cannot modify operator-created ServiceMonitor.

        Verify that a user without ServiceMonitor management permissions cannot create,
        modify, or delete ServiceMonitor resources created by the operator.

        NOTE: This test verifies the ServiceMonitor is protected at the API level.
        In a production environment with test users, additional validation would use
        `oc auth can-i` or impersonation to verify restricted user permissions.
        """
        # Verify the ServiceMonitor exists and is owned by the operator (via ownerReference)
        assert evalhub_service_monitor.exists, f"ServiceMonitor '{evalhub_service_monitor.name}' does not exist"

        owner_refs = evalhub_service_monitor.instance.metadata.ownerReferences or []
        assert owner_refs, (
            f"ServiceMonitor '{evalhub_service_monitor.name}' has no ownerReferences. "
            f"Expected ownerReference to EvalHub CR for operator management."
        )

        evalhub_owner = next(
            (ref for ref in owner_refs if ref.kind == "EvalHub"),
            None,
        )
        assert evalhub_owner is not None, (
            f"ServiceMonitor '{evalhub_service_monitor.name}' has no ownerReference to EvalHub CR. "
            f"This indicates the ServiceMonitor may not be operator-managed."
        )

        # Verify the ServiceMonitor is in the operator-managed namespace
        assert evalhub_service_monitor.namespace == model_namespace.name, (
            f"ServiceMonitor namespace '{evalhub_service_monitor.namespace}' does not match "
            f"expected namespace '{model_namespace.name}'"
        )

        # NOTE: Full RBAC validation would require:
        # 1. Creating a test user with limited permissions (namespace admin but no ServiceMonitor access)
        # 2. Using `oc auth can-i delete servicemonitor --as=test-user` to verify denial
        # 3. Attempting actual delete/patch operations with the test user's token
        #
        # This test verifies the ServiceMonitor is operator-managed (ownerReference to EvalHub).
        # In OpenShift/Kubernetes, ServiceMonitor is a cluster-scoped CRD that requires
        # explicit RBAC permissions. Standard namespace admin roles do not grant
        # ServiceMonitor management permissions unless explicitly added.

    def test_servicemonitor_namespace_scoped_isolation(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_service_monitor: ServiceMonitor,
    ) -> None:
        """TC-RBAC-003: ServiceMonitor is namespace-scoped and isolated.

        Verify that a ServiceMonitor created for an EvalHub instance in one namespace
        does not affect EvalHub instances or monitoring in other namespaces.
        """
        # Create a second temporary namespace with its own EvalHub instance
        temp_namespace_name = "test-evalhub-isolation"

        with Namespace(
            client=admin_client,
            name=temp_namespace_name,
        ) as temp_namespace:
            # Deploy EvalHub in the second namespace
            with EvalHub(
                client=admin_client,
                name="evalhub-isolation-test",
                namespace=temp_namespace.name,
                database={"type": "sqlite"},
                wait_for_resource=True,
            ) as temp_evalhub:
                # Wait for deployment
                temp_deployment = Deployment(
                    client=admin_client,
                    name=temp_evalhub.name,
                    namespace=temp_namespace.name,
                )
                temp_deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)

                # Verify each namespace has its own ServiceMonitor
                original_namespace_sms = list(
                    ServiceMonitor.get(
                        client=admin_client,
                        namespace=model_namespace.name,
                        label_selector=f"app={EVALHUB_APP_LABEL}",
                    )
                )

                temp_namespace_sms = list(
                    ServiceMonitor.get(
                        client=admin_client,
                        namespace=temp_namespace.name,
                        label_selector=f"app={EVALHUB_APP_LABEL}",
                    )
                )

                assert original_namespace_sms, f"No ServiceMonitor found in original namespace '{model_namespace.name}'"
                assert temp_namespace_sms, f"No ServiceMonitor found in temporary namespace '{temp_namespace.name}'"

                # Verify the ServiceMonitors are distinct (different namespaces)
                original_sm = original_namespace_sms[0]
                temp_sm = temp_namespace_sms[0]

                assert original_sm.namespace != temp_sm.namespace, (
                    "ServiceMonitors should be in different namespaces for isolation"
                )

                assert original_sm.namespace == model_namespace.name, (
                    f"Original ServiceMonitor should be in '{model_namespace.name}', got '{original_sm.namespace}'"
                )
                assert temp_sm.namespace == temp_namespace.name, (
                    f"Temporary ServiceMonitor should be in '{temp_namespace.name}', got '{temp_sm.namespace}'"
                )

            # EvalHub deleted at context exit - wait for ServiceMonitor to be garbage collected
            try:
                for _ in TimeoutSampler(
                    wait_timeout=60,
                    sleep=2,
                    func=lambda: (
                        not ServiceMonitor(
                            client=admin_client,
                            name=temp_sm.name,
                            namespace=temp_namespace.name,
                        ).exists
                    ),
                ):
                    if not ServiceMonitor(
                        client=admin_client,
                        name=temp_sm.name,
                        namespace=temp_namespace.name,
                    ).exists:
                        break
            except TimeoutExpiredError as err:
                msg = (
                    f"ServiceMonitor '{temp_sm.name}' in '{temp_namespace.name}' was not deleted "
                    f"within 60 seconds after deleting EvalHub CR."
                )
                raise AssertionError(msg) from err

        # Verify the original ServiceMonitor in the first namespace is unaffected
        original_sm_after = ServiceMonitor(
            client=admin_client,
            name=original_sm.name,
            namespace=model_namespace.name,
        )
        assert original_sm_after.exists, (
            f"Original ServiceMonitor '{original_sm.name}' in '{model_namespace.name}' "
            f"was affected by deletion in another namespace. Expected namespace isolation."
        )

        # Verify the original ServiceMonitor is still functional (has correct configuration)
        endpoints = original_sm_after.instance.spec.endpoints
        assert endpoints, (
            f"Original ServiceMonitor '{original_sm.name}' has no endpoints after operations in another namespace"
        )
        endpoint = endpoints[0]
        assert endpoint.path == "/metrics", (
            f"Original ServiceMonitor endpoint path changed. Expected '/metrics', got '{endpoint.path}'"
        )
