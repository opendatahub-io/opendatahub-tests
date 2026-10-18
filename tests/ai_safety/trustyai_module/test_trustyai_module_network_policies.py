import pytest
from ocp_resources.deployment import Deployment
from ocp_resources.network_policy import NetworkPolicy

from tests.ai_safety.trustyai_module.constants import (
    TRUSTYAI_MODULE_OPERATOR_IDENTITY_LABEL,
    TRUSTYAI_MODULE_OPERATOR_PROBE_PORT,
)
from utilities.network_policy import (
    get_network_policy_egress_rules,
    get_network_policy_ingress_rules,
    get_network_policy_pod_selector_labels,
    get_network_policy_policy_types,
)


def _assert_policy_selects_module_operator_pods(network_policy: NetworkPolicy, deployment: Deployment) -> None:
    """Assert that a NetworkPolicy pod selector is pinned to the module operator pod identity.

    The kubebuilder default control-plane: controller-manager label is shared by every module
    operator deployed in the applications namespace, so the module-owned
    app.kubernetes.io/name label is the distinguishing pod identity.

    Args:
        network_policy: Module operator NetworkPolicy to check.
        deployment: Module operator Deployment the policy must apply to.
    """
    pod_selector = get_network_policy_pod_selector_labels(network_policy=network_policy)
    pod_labels = deployment.instance.to_dict()["spec"]["template"]["metadata"]["labels"]
    assert pod_selector, f"{network_policy.name} must not select every pod in the namespace"
    assert TRUSTYAI_MODULE_OPERATOR_IDENTITY_LABEL.items() <= pod_selector.items(), (
        f"{network_policy.name} must select pods through the module operator identity label: selector={pod_selector}"
    )
    assert pod_selector.items() <= pod_labels.items(), (
        f"{network_policy.name} must not select pods other than {deployment.name}: "
        f"selector={pod_selector}, deployment pod labels={pod_labels}"
    )


@pytest.mark.ai_safety
@pytest.mark.usefixtures("modular_trustyai_architecture")
class TestTrustyAIModuleNetworkPolicies:
    """Validate the NetworkPolicy deployed for the TrustyAI module operator manager pod.

    The platform operator renders the module bundle (trustyai-operator-module/config) with the
    module operator Deployment, including its module-bundle NetworkPolicy (trustyai-service-operator
    PR #1035): a single policy enforcing both directions, admitting only the health/readiness
    probe port (TCP 8081, port-scoped because node-originated probes cannot be constrained by
    peer selectors) and declaring the explicit allow-all egress posture required by
    ODH-ADR-Operator-0016.
    """

    @pytest.mark.tier1
    def test_module_operator_network_policy_exists(
        self,
        trustyai_module_operator_network_policy: NetworkPolicy,
    ) -> None:
        """Given the TrustyAI module operator is deployed
        When looking for the NetworkPolicy protecting its manager pod
        Then the module-bundle policy is deployed in the applications namespace
        """
        network_policy = trustyai_module_operator_network_policy
        assert network_policy.exists, (
            f"NetworkPolicy {network_policy.name} must be deployed in namespace {network_policy.namespace}"
        )

    @pytest.mark.tier2
    def test_network_policy_selects_only_module_operator_pods(
        self,
        trustyai_module_operator_deployment: Deployment,
        trustyai_module_operator_network_policy: NetworkPolicy,
    ) -> None:
        """Given the TrustyAI module operator is deployed
        When inspecting the pod selector of its NetworkPolicy
        Then it selects only the module operator pods through the module identity label
        """
        _assert_policy_selects_module_operator_pods(
            network_policy=trustyai_module_operator_network_policy,
            deployment=trustyai_module_operator_deployment,
        )

    @pytest.mark.tier2
    def test_ingress_admits_probe_port_only(
        self,
        trustyai_module_operator_network_policy: NetworkPolicy,
    ) -> None:
        """Given the module operator NetworkPolicy
        When inspecting its policyTypes and ingress rules
        Then it enforces both directions in a single policy and admits only the health/readiness
        probe port, port-scoped because node-originated probes cannot be peer-restricted
        """
        network_policy = trustyai_module_operator_network_policy
        assert get_network_policy_policy_types(network_policy=network_policy) == ["Ingress", "Egress"], (
            f"{network_policy.name} must enforce Ingress and Egress in a single policy"
        )

        ingress_rules = get_network_policy_ingress_rules(network_policy=network_policy)
        assert len(ingress_rules) == 1, f"{network_policy.name} must carry exactly one ingress rule: {ingress_rules}"
        probe_rule = ingress_rules[0]
        assert probe_rule == {"ports": [{"protocol": "TCP", "port": TRUSTYAI_MODULE_OPERATOR_PROBE_PORT}]}, (
            f"The ingress rule must admit only TCP {TRUSTYAI_MODULE_OPERATOR_PROBE_PORT}: {probe_rule}"
        )

    @pytest.mark.tier2
    def test_egress_declares_explicit_allow_all(
        self,
        trustyai_module_operator_network_policy: NetworkPolicy,
    ) -> None:
        """Given the module operator NetworkPolicy
        When inspecting its egress rules
        Then it carries a single explicit allow-all rule that satisfies the platform
        explicit-egress requirement for the module operator pod
        """
        network_policy = trustyai_module_operator_network_policy
        assert get_network_policy_egress_rules(network_policy=network_policy) == [{}], (
            f"{network_policy.name} must carry exactly one explicit allow-all egress rule"
        )
