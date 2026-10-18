import json
from typing import Any

import pytest
from ocp_resources.deployment import Deployment
from ocp_resources.network_policy import NetworkPolicy

from tests.ai_safety.trustyai_operator.constants import (
    OPENSHIFT_MONITORING_PEER,
    OPENSHIFT_USER_WORKLOAD_MONITORING_PEER,
    TRUSTYAI_OPERATOR_IDENTITY_LABEL,
    TRUSTYAI_OPERATOR_METRICS_PORT,
    TRUSTYAI_OPERATOR_PROBE_PORT,
    TRUSTYAI_OPERATOR_WEBHOOK_PORT,
)
from utilities.network_policy import (
    get_network_policy_egress_rules,
    get_network_policy_ingress_rules,
    get_network_policy_pod_selector_labels,
    get_network_policy_policy_types,
)

EXPECTED_METRICS_PEERS: tuple[dict[str, Any], ...] = (
    OPENSHIFT_MONITORING_PEER,
    OPENSHIFT_USER_WORKLOAD_MONITORING_PEER,
)
EXPECTED_INGRESS_PORTS: tuple[int, ...] = (
    TRUSTYAI_OPERATOR_METRICS_PORT,
    TRUSTYAI_OPERATOR_PROBE_PORT,
    TRUSTYAI_OPERATOR_WEBHOOK_PORT,
)


def _assert_policy_selects_operator_pods(network_policy: NetworkPolicy, deployment: Deployment) -> None:
    """Assert that a NetworkPolicy pod selector is pinned to the workload operator pod identity.

    Args:
        network_policy: Workload operator NetworkPolicy to check.
        deployment: Workload operator Deployment the policy must apply to.
    """
    pod_selector = get_network_policy_pod_selector_labels(network_policy=network_policy)
    pod_labels = deployment.instance.to_dict()["spec"]["template"]["metadata"]["labels"]
    assert pod_selector, f"{network_policy.name} must not select every pod in the namespace"
    assert TRUSTYAI_OPERATOR_IDENTITY_LABEL.items() <= pod_selector.items(), (
        f"{network_policy.name} must select pods through the workload operator identity label: selector={pod_selector}"
    )
    assert pod_selector.items() <= pod_labels.items(), (
        f"{network_policy.name} must not select pods other than {deployment.name}: "
        f"selector={pod_selector}, deployment pod labels={pod_labels}"
    )


@pytest.mark.ai_safety
@pytest.mark.usefixtures("trustyai_operator_installed")
class TestTrustyAIOperatorNetworkPolicies:
    """Validate the NetworkPolicies deployed for the TrustyAI workload operator manager pod.

    The workload operator (trustyai-service-operator) manager pod is deployed with two
    direction-split policies (trustyai-service-operator PR #1035): an ingress-only policy
    admitting the secure-metrics, health-probe and CRD-conversion-webhook ports, and a
    dedicated policy declaring the operator's explicit allow-all egress posture required
    by ODH-ADR-Operator-0016.
    """

    @pytest.mark.tier1
    def test_network_policies_exist(
        self,
        trustyai_operator_ingress_network_policy: NetworkPolicy,
        trustyai_operator_egress_network_policy: NetworkPolicy,
    ) -> None:
        """Given the TrustyAI workload operator is deployed
        When looking for the NetworkPolicies protecting its manager pod
        Then both direction-split policies are deployed in the applications namespace
        """
        for network_policy in (
            trustyai_operator_ingress_network_policy,
            trustyai_operator_egress_network_policy,
        ):
            assert network_policy.exists, (
                f"NetworkPolicy {network_policy.name} must be deployed in namespace {network_policy.namespace}"
            )

    @pytest.mark.tier2
    def test_network_policies_select_only_operator_pods(
        self,
        trustyai_operator_deployment: Deployment,
        trustyai_operator_ingress_network_policy: NetworkPolicy,
        trustyai_operator_egress_network_policy: NetworkPolicy,
    ) -> None:
        """Given the TrustyAI workload operator is deployed
        When inspecting the pod selectors of the operator NetworkPolicies
        Then both policies select only the workload operator pods through the operator identity label
        """
        for network_policy in (
            trustyai_operator_ingress_network_policy,
            trustyai_operator_egress_network_policy,
        ):
            _assert_policy_selects_operator_pods(
                network_policy=network_policy,
                deployment=trustyai_operator_deployment,
            )

    @pytest.mark.tier2
    def test_ingress_policy_admits_expected_ports_only(
        self,
        trustyai_operator_ingress_network_policy: NetworkPolicy,
    ) -> None:
        """Given the workload operator ingress NetworkPolicy
        When inspecting its policyTypes and ingress rules
        Then it enforces Ingress only and admits the secure-metrics port from the supported OpenShift
        monitoring namespaces, while the health-probe and webhook ports stay port-scoped
        """
        network_policy = trustyai_operator_ingress_network_policy
        assert get_network_policy_policy_types(network_policy=network_policy) == ["Ingress"], (
            f"{network_policy.name} must be ingress-only; the operator egress posture is declared by "
            f"the dedicated allow-egress policy"
        )

        ingress_rules = get_network_policy_ingress_rules(network_policy=network_policy)
        rules_by_port: dict[int, dict[str, Any]] = {}
        for rule in ingress_rules:
            ports = rule.get("ports", [])
            assert len(ports) == 1, f"Each ingress rule must scope exactly one port: {rule}"
            rules_by_port[ports[0]["port"]] = rule

        assert set(rules_by_port) == set(EXPECTED_INGRESS_PORTS), (
            f"{network_policy.name} must admit exactly the ports {EXPECTED_INGRESS_PORTS}: got {sorted(rules_by_port)}"
        )
        for port, rule in rules_by_port.items():
            assert rule["ports"] == [{"protocol": "TCP", "port": port}], f"Port {port} must be TCP only: {rule}"

        metrics_rule = rules_by_port[TRUSTYAI_OPERATOR_METRICS_PORT]
        assert metrics_rule.get("from"), f"The metrics port rule must restrict its sources: {metrics_rule}"
        assert sorted(json.dumps(peer, sort_keys=True) for peer in metrics_rule["from"]) == sorted(
            json.dumps(peer, sort_keys=True) for peer in EXPECTED_METRICS_PEERS
        ), (
            f"Port {TRUSTYAI_OPERATOR_METRICS_PORT} must be reachable only from the supported "
            f"OpenShift monitoring namespaces: {metrics_rule}"
        )

        for port in (TRUSTYAI_OPERATOR_PROBE_PORT, TRUSTYAI_OPERATOR_WEBHOOK_PORT):
            assert "from" not in rules_by_port[port], f"Port {port} must stay port-scoped: {rules_by_port[port]}"

    @pytest.mark.tier2
    def test_egress_policy_declares_explicit_allow_all(
        self,
        trustyai_operator_egress_network_policy: NetworkPolicy,
    ) -> None:
        """Given the workload operator egress NetworkPolicy
        When inspecting its policyTypes and egress rules
        Then it declares Egress only, with a single explicit allow-all rule that satisfies the platform
        explicit-egress requirement for the operator pod
        """
        network_policy = trustyai_operator_egress_network_policy
        assert get_network_policy_policy_types(network_policy=network_policy) == ["Egress"], (
            f"{network_policy.name} must declare Egress only"
        )
        assert get_network_policy_egress_rules(network_policy=network_policy) == [{}], (
            f"{network_policy.name} must carry exactly one explicit allow-all egress rule"
        )
