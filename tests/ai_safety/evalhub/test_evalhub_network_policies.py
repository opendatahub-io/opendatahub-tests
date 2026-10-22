import json
from collections.abc import Callable
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.evalhub import EvalHub
from ocp_resources.namespace import Namespace
from ocp_resources.network_policy import NetworkPolicy
from ocp_resources.route import Route
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutExpiredError

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
    EVALHUB_NP_OPERATOR_POD_LABELS,
    EVALHUB_NP_POLICY_GROUP_LABEL,
    EVALHUB_NP_PROMETHEUS_POD_LABELS,
    EVALHUB_NP_ROLE,
    EVALHUB_NP_ROUTER_POLICY_GROUP,
    EVALHUB_SERVICE_PORT,
    EVALHUB_TENANT_LABEL_KEY,
)
from tests.ai_safety.evalhub.k8s_lifecycle_signals.constants import (
    LIFECYCLE_REASON_FAILED,
    LIFECYCLE_SOURCE_OPERATOR,
)
from tests.ai_safety.evalhub.k8s_lifecycle_signals.utils import (
    wait_for_event_from_component,
)
from tests.ai_safety.evalhub.utils import (
    wait_for_evalhub_job,
    wait_for_evalhub_runtime_job_count,
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
        The five expected ingress rules: OpenShift router, Dashboard eval-hub-ui proxy,
        the operator manager Pod (failure-event reporter), evaluation-job callbacks,
        and Prometheus metrics scraping.
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
                {
                    "namespaceSelector": {
                        "matchLabels": {EVALHUB_NP_DASHBOARD_NAMESPACE_LABEL: applications_namespace}
                    },
                    "podSelector": {"matchLabels": EVALHUB_NP_OPERATOR_POD_LABELS},
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
    router (API Route), the ODH Dashboard eval-hub-ui proxy, the operator's own manager
    Pod (which reports job failures the Job containers cannot) and evaluation-job
    callback pods reach kube-rbac-proxy on TCP 8443, while Prometheus scrapes the
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
            f"(router, Dashboard, operator failure-event reporter, job callbacks, Prometheus): {ingress_rules}"
        )
        # Ingress rule order carries no meaning; compare as sorted multisets.
        assert sorted(ingress_rules, key=_canonical_rule) == sorted(expected_rules, key=_canonical_rule), (
            f"{evalhub_api_network_policy.name} must admit exactly the expected peers: {ingress_rules}"
        )

    @pytest.mark.tier2
    def test_api_network_policy_admits_operator_failure_event_peer(
        self,
        evalhub_api_network_policy: NetworkPolicy,
        trustyai_operator_deployment: Deployment,
    ) -> None:
        """The operator manager Pod may POST evaluation-job failure events to the EvalHub API.

        The operator's evalhub-evaluation-job-failure reconciler (and the Kueue Workload
        failure reconciler) POSTs /api/v1/evaluations/jobs/<id>/events when a Job pod
        never starts: an unschedulable Pod or missing PVC, or a failed init container
        for a bad git ref/sub_path or an invalid Hugging Face repo/sub_path. Without an
        ingress peer for the operator Pod the operator-generated policy drops those
        POSTs, EvalHub never records the failure, and the job stays pending until the
        caller times out instead of moving to failed.
        """
        applications_namespace = py_config["applications_namespace"]
        ingress_rules = get_network_policy_ingress_rules(network_policy=evalhub_api_network_policy)

        def _is_operator_failure_event_peer(peer: dict[str, Any]) -> bool:
            return (
                peer.get("namespaceSelector", {}).get("matchLabels", {}).get(EVALHUB_NP_DASHBOARD_NAMESPACE_LABEL)
                == applications_namespace
                and peer.get("podSelector", {}).get("matchLabels") == EVALHUB_NP_OPERATOR_POD_LABELS
            )

        operator_rules = [
            rule
            for rule in ingress_rules
            if any(_is_operator_failure_event_peer(peer) for peer in rule.get("from", []))
        ]
        assert operator_rules, (
            f"{evalhub_api_network_policy.name} must admit the operator manager Pod "
            f"({EVALHUB_NP_OPERATOR_POD_LABELS}) in {applications_namespace} on TCP {EVALHUB_SERVICE_PORT}; "
            f"without it the operator's job failure events are dropped: {ingress_rules}"
        )
        assert operator_rules[0]["ports"] == [_tcp_port(EVALHUB_SERVICE_PORT)], (
            f"{evalhub_api_network_policy.name} must admit the operator failure-event peer on TCP "
            f"{EVALHUB_SERVICE_PORT}: {operator_rules[0]['ports']}"
        )

        # The peer must select the Pods that actually run the failure reconcilers, not a
        # stale identity, so bind the selector to the deployed manager Deployment labels.
        operator_pod_labels = trustyai_operator_deployment.instance.to_dict()["spec"]["template"]["metadata"]["labels"]
        assert EVALHUB_NP_OPERATOR_POD_LABELS.items() <= operator_pod_labels.items(), (
            f"{evalhub_api_network_policy.name} selects {EVALHUB_NP_OPERATOR_POD_LABELS}, which does not match the "
            f"deployed {trustyai_operator_deployment.name} pod labels {operator_pod_labels}"
        )


# The operator's failure watcher waits out this grace before treating an
# unschedulable Pod as a failure (trustyai-service-operator
# controllers/evalhub/evaluation_job_failure_reconciler.go), so the end-to-end
# path needs more than a label read.
OPERATOR_FAILURE_EVENT_TIMEOUT: int = 600


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-np-failure-event"},
        ),
    ],
    indirect=True,
)
@pytest.mark.ai_safety
@pytest.mark.usefixtures("evalhub_operator_installed")
class TestEvalHubApiPolicyOperatorFailureEventPath:
    """End-to-end regression coverage for the operator's own EvalHub API client path.

    The API NetworkPolicy has to admit the operator manager Pod because that Pod
    reports the failures the evaluation Job containers cannot report. This class
    exercises the consequence rather than the policy shape: a Job whose Pod can
    never be scheduled must still end up failed, and it must get there through an
    EvaluationFailed Event emitted by the operator, which the operator records only
    after its POST to the EvalHub API succeeds. With the peer missing, the policy
    silently drops that POST, no Event appears, and the Job stays pending.
    """

    @pytest.mark.tier2
    def test_unschedulable_job_fails_through_operator_failure_event(
        self,
        admin_client: DynamicClient,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        submit_pvc_job: Callable[..., str],
    ) -> None:
        """A Job referencing a PVC that does not exist still reaches failed.

        Given an evaluation Job whose Pod can never be scheduled because its PVC
        does not exist,
        when the operator's failure watcher passes its scheduling grace,
        then the operator POSTs the failure to the EvalHub API, the batch Job gets an
        EvaluationFailed Event from source.component=trustyai-service-operator, and
        the EvalHub job state becomes failed instead of staying pending.
        """
        claim_name = "nonexistent-pvc-operator-failure-event"
        job_id = submit_pvc_job(
            claim_name=claim_name,
            job_name="np-operator-failure-event",
        )
        batch_jobs = wait_for_evalhub_runtime_job_count(
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
            evalhub_job_id=job_id,
            minimum=1,
        )
        batch_job_name = batch_jobs[0].name

        try:
            failure_event = wait_for_event_from_component(
                admin_client=admin_client,
                job_name=batch_job_name,
                namespace=tenant_a_namespace.name,
                reason=LIFECYCLE_REASON_FAILED,
                source_component=LIFECYCLE_SOURCE_OPERATOR,
                timeout=OPERATOR_FAILURE_EVENT_TIMEOUT,
            )
        except TimeoutExpiredError:
            pytest.fail(
                f"No {LIFECYCLE_REASON_FAILED} Event from {LIFECYCLE_SOURCE_OPERATOR} on batch Job "
                f"{batch_job_name} within {OPERATOR_FAILURE_EVENT_TIMEOUT}s. The operator detects the "
                f"unschedulable Pod but its failure event never reaches EvalHub when the API "
                f"NetworkPolicy does not admit the operator manager Pod "
                f"({EVALHUB_NP_OPERATOR_POD_LABELS}); the evaluation job then stays pending forever."
            )

        event_message = str(failure_event.get("message", ""))
        assert claim_name in event_message, (
            f"Operator failure event does not carry the missing-PVC detail for {claim_name}: {event_message!r}"
        )

        job_data = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
            timeout=OPERATOR_FAILURE_EVENT_TIMEOUT,
        )
        state = job_data.get("status", {}).get("state")
        assert state == "failed", (
            f"Evaluation Job with an unschedulable Pod must reach 'failed', got {state!r}. "
            f"A 'pending' job means the operator's failure event did not reach EvalHub: {job_data.get('status')}"
        )
