import json
from typing import Any

import pytest
from ocp_resources.deployment import Deployment
from ocp_resources.evalhub import EvalHub
from ocp_resources.network_policy import NetworkPolicy
from pytest_testconfig import config as py_config

from tests.ai_safety.evalhub.constants import (
    EVALHUB_K8S_LABEL_APP,
    EVALHUB_K8S_LABEL_APP_VALUE,
    EVALHUB_K8S_LABEL_COMPONENT,
    EVALHUB_K8S_LABEL_COMPONENT_VALUE,
    EVALHUB_METRICS_PORT,
    EVALHUB_NP_COMPONENT,
    EVALHUB_NP_DASHBOARD_NAMESPACE_LABEL,
    EVALHUB_NP_DASHBOARD_POD_LABELS,
    EVALHUB_NP_INSTANCE_NAME_LABEL,
    EVALHUB_NP_INSTANCE_NAMESPACE_LABEL,
    EVALHUB_NP_MONITORING_POLICY_GROUP,
    EVALHUB_NP_POLICY_GROUP_LABEL,
    EVALHUB_NP_PROMETHEUS_POD_LABELS,
    EVALHUB_NP_ROLE,
    EVALHUB_NP_ROUTER_POLICY_GROUP,
    EVALHUB_SERVICE_PORT,
    EVALHUB_TENANT_LABEL_KEY,
)
from utilities.network_policy import (
    NETWORK_POLICY_COMPONENT_LABEL,
    NETWORK_POLICY_MANAGED_BY_LABEL,
    NETWORK_POLICY_MANAGER,
    NETWORK_POLICY_OWNER_UID_LABEL,
    NETWORK_POLICY_ROLE_LABEL,
    get_network_policy_ingress_rules,
    get_network_policy_labels,
    get_network_policy_pod_selector_labels,
    get_network_policy_policy_types,
)


def _tcp_port(port: int) -> dict[str, Any]:
    return {"port": port, "protocol": "TCP"}


def _expected_ingress_rules(evalhub_cr: EvalHub, applications_namespace: str) -> list[dict[str, Any]]:
    """Build the exact ingress rules the operator must reconcile for the EvalHub API policy.

    Args:
        evalhub_cr: EvalHub CR the policy belongs to.
        applications_namespace: ODH applications namespace hosting the Dashboard eval-hub-ui module.

    Returns:
        The four expected ingress rules: OpenShift router, Dashboard eval-hub-ui proxy,
        evaluation-job callbacks, and Prometheus metrics scraping.
    """
    job_pod_labels = {
        EVALHUB_K8S_LABEL_APP: EVALHUB_K8S_LABEL_APP_VALUE,
        EVALHUB_K8S_LABEL_COMPONENT: EVALHUB_K8S_LABEL_COMPONENT_VALUE,
        EVALHUB_NP_INSTANCE_NAME_LABEL: evalhub_cr.name,
        EVALHUB_NP_INSTANCE_NAMESPACE_LABEL: evalhub_cr.namespace,
    }
    return [
        {
            "from": [
                {"namespaceSelector": {"matchLabels": {EVALHUB_NP_POLICY_GROUP_LABEL: EVALHUB_NP_ROUTER_POLICY_GROUP}}}
            ],
            "ports": [_tcp_port(EVALHUB_SERVICE_PORT)],
        },
        {
            "from": [
                {
                    "namespaceSelector": {
                        "matchLabels": {EVALHUB_NP_DASHBOARD_NAMESPACE_LABEL: applications_namespace}
                    },
                    "podSelector": {"matchLabels": EVALHUB_NP_DASHBOARD_POD_LABELS},
                }
            ],
            "ports": [_tcp_port(EVALHUB_SERVICE_PORT)],
        },
        {
            "from": [
                {"podSelector": {"matchLabels": job_pod_labels}},
                {
                    "namespaceSelector": {
                        "matchExpressions": [{"key": EVALHUB_TENANT_LABEL_KEY, "operator": "Exists"}]
                    },
                    "podSelector": {"matchLabels": job_pod_labels},
                },
            ],
            "ports": [_tcp_port(EVALHUB_SERVICE_PORT)],
        },
        {
            "from": [
                {
                    "namespaceSelector": {
                        "matchLabels": {EVALHUB_NP_POLICY_GROUP_LABEL: EVALHUB_NP_MONITORING_POLICY_GROUP}
                    },
                    "podSelector": {"matchLabels": EVALHUB_NP_PROMETHEUS_POD_LABELS},
                }
            ],
            "ports": [_tcp_port(EVALHUB_METRICS_PORT)],
        },
    ]


def _canonical_rule(rule: dict[str, Any]) -> str:
    return json.dumps(rule, sort_keys=True)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-np"},
        ),
    ],
    indirect=True,
)
@pytest.mark.ai_safety
@pytest.mark.usefixtures("evalhub_operator_installed")
class TestEvalHubNetworkPolicies:
    """Validate the operator-managed NetworkPolicy protecting the EvalHub API pod.

    The TrustyAI workload operator reconciles one ingress-only policy per EvalHub CR
    (trustyai-service-operator controllers/evalhub/network_policy.go): the OpenShift
    router (API Route), the ODH Dashboard eval-hub-ui proxy and evaluation-job callback
    pods reach kube-rbac-proxy on TCP 8443, while Prometheus scrapes the
    unauthenticated metrics listener on TCP 8081. The plain sqlite EvalHub CR these
    tests create has MCP disabled, so no MCP ingress rule is expected.
    """

    @pytest.mark.tier1
    def test_api_network_policy_exists(self, evalhub_api_network_policy: NetworkPolicy) -> None:
        """Given a deployed EvalHub CR, the operator reconciles an API NetworkPolicy for it."""
        assert evalhub_api_network_policy.exists, (
            f"EvalHub API NetworkPolicy missing in {evalhub_api_network_policy.namespace}"
        )

    @pytest.mark.tier2
    def test_api_network_policy_selects_only_api_pods(
        self,
        evalhub_api_network_policy: NetworkPolicy,
        evalhub_deployment: Deployment,
        evalhub_cr: EvalHub,
    ) -> None:
        """The policy selector is pinned to this CR's API pod identity and nothing else."""
        pod_selector = get_network_policy_pod_selector_labels(network_policy=evalhub_api_network_policy)
        policy_labels = get_network_policy_labels(network_policy=evalhub_api_network_policy)
        pod_labels = evalhub_deployment.instance.to_dict()["spec"]["template"]["metadata"]["labels"]
        owner_uid = evalhub_cr.instance.to_dict()["metadata"]["uid"]

        assert pod_selector, f"{evalhub_api_network_policy.name} must not select every pod in the namespace"
        assert pod_selector == {
            NETWORK_POLICY_COMPONENT_LABEL: EVALHUB_NP_COMPONENT,
            NETWORK_POLICY_ROLE_LABEL: EVALHUB_NP_ROLE,
            NETWORK_POLICY_OWNER_UID_LABEL: owner_uid,
        }, f"{evalhub_api_network_policy.name} selector must be the EvalHub API identity: {pod_selector}"
        assert policy_labels.get(NETWORK_POLICY_MANAGED_BY_LABEL) == NETWORK_POLICY_MANAGER, (
            f"{evalhub_api_network_policy.name} must be operator-managed: {policy_labels}"
        )
        assert pod_selector.items() <= pod_labels.items(), (
            f"{evalhub_api_network_policy.name} must not select pods other than {evalhub_deployment.name}: "
            f"selector={pod_selector}, deployment pod labels={pod_labels}"
        )

    @pytest.mark.tier2
    def test_api_network_policy_is_ingress_only(self, evalhub_api_network_policy: NetworkPolicy) -> None:
        """The API policy restricts ingress only; egress stays unrestricted."""
        policy_types = get_network_policy_policy_types(network_policy=evalhub_api_network_policy)
        assert policy_types == ["Ingress"], (
            f"{evalhub_api_network_policy.name} must be ingress-only: policyTypes={policy_types}"
        )

    @pytest.mark.tier2
    def test_api_network_policy_admits_expected_peers_only(
        self,
        evalhub_api_network_policy: NetworkPolicy,
        evalhub_cr: EvalHub,
    ) -> None:
        """Ingress admits exactly the router, Dashboard, evaluation-job callback and Prometheus peers."""
        ingress_rules = get_network_policy_ingress_rules(network_policy=evalhub_api_network_policy)
        expected_rules = _expected_ingress_rules(
            evalhub_cr=evalhub_cr,
            applications_namespace=py_config["applications_namespace"],
        )
        assert len(ingress_rules) == len(expected_rules), (
            f"{evalhub_api_network_policy.name} must admit exactly {len(expected_rules)} ingress rules "
            f"(router, Dashboard, job callbacks, Prometheus): {ingress_rules}"
        )
        # Ingress rule order carries no meaning; compare as sorted multisets.
        assert sorted(ingress_rules, key=_canonical_rule) == sorted(expected_rules, key=_canonical_rule), (
            f"{evalhub_api_network_policy.name} must admit exactly the expected peers: {ingress_rules}"
        )
