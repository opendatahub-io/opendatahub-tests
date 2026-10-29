from typing import Any

TRUSTYAI_OPERATOR_DEPLOYMENT: str = "trustyai-service-operator-controller-manager"

TRUSTYAI_OPERATOR_INGRESS_NETWORK_POLICY: str = "trustyai-service-operator-controller-manager-allow-ports"
TRUSTYAI_OPERATOR_EGRESS_NETWORK_POLICY: str = "trustyai-service-operator-controller-manager-allow-egress"
TRUSTYAI_OPERATOR_IDENTITY_LABEL: dict[str, str] = {"control-plane": "trustyai-service-operator"}

TRUSTYAI_OPERATOR_METRICS_PORT: int = 8443
TRUSTYAI_OPERATOR_PROBE_PORT: int = 8081
TRUSTYAI_OPERATOR_WEBHOOK_PORT: int = 9443

OPENSHIFT_MONITORING_PEER: dict[str, Any] = {
    "namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "openshift-monitoring"}}
}
OPENSHIFT_USER_WORKLOAD_MONITORING_PEER: dict[str, Any] = {
    "namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "openshift-user-workload-monitoring"}}
}
