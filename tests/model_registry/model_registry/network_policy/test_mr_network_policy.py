import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.network_policy import NetworkPolicy

from utilities.resources.model_registry_modelregistry_opendatahub_io import ModelRegistry
from utilities.resources.pod import Pod

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.parametrize(
    "model_registry_metadata_db_resources, model_registry_instance",
    [pytest.param({}, {}, id="test_mr_network_policy")],
    indirect=True,
)
@pytest.mark.usefixtures("updated_dsc_component_state_scope_session")
class TestModelRegistryNetworkPolicy:
    """Validate that model registry https-route NetworkPolicy allows dashboard namespace traffic."""

    def test_https_route_ingress_namespace_labels(
        self,
        admin_client: DynamicClient,
        model_registry_instance: list[ModelRegistry],
        model_registry_namespace: str,
    ) -> None:
        """Given a deployed model registry instance
        When inspecting its https-route NetworkPolicy ingress namespace selectors
        Then it allows traffic from dashboard namespace and OpenShift ingress namespace
        """
        np_name = f"{model_registry_instance[0].name}-https-route"
        network_policy = NetworkPolicy(
            client=admin_client,
            name=np_name,
            namespace=model_registry_namespace,
            ensure_exists=True,
        )
        from_rules = network_policy.instance.spec.ingress[0]["from"]
        namespace_labels = [
            rule.namespaceSelector.matchLabels
            for rule in from_rules
            if hasattr(rule, "namespaceSelector") and rule.namespaceSelector
        ]
        assert any(labels.get("opendatahub.io/generated-namespace") == "true" for labels in namespace_labels), (
            f"{np_name} should allow traffic from dashboard namespace (opendatahub.io/generated-namespace: true)"
        )
        assert any(labels.get("network.openshift.io/policy-group") == "ingress" for labels in namespace_labels), (
            f"{np_name} should allow traffic from OpenShift ingress namespace"
        )

    def test_dashboard_can_reach_model_registry(
        self,
        dashboard_pod: Pod,
        model_registry_instance: list[ModelRegistry],
        model_registry_namespace: str,
    ) -> None:
        """Given a dashboard pod in the applications namespace
        When curling the model registry internal service on the kube-rbac-proxy port
        Then the connection is not blocked by the NetworkPolicy
        """
        mr_name = model_registry_instance[0].name
        service_url = f"https://{mr_name}.{model_registry_namespace}.svc.cluster.local:8443"
        result = dashboard_pod.execute(command=["curl", "-k", "--connect-timeout", "10", service_url])
        LOGGER.info(f"curl to {service_url}: rc={result.rc}, stdout={result.stdout}, stderr={result.stderr}")
        assert result.rc == 0, (
            f"Dashboard pod cannot reach model registry at {service_url} — "
            f"NetworkPolicy may be blocking traffic (rc={result.rc}, stderr={result.stderr})"
        )
        assert "Connection timed out" not in result.stdout, (
            f"Dashboard pod connection timed out to model registry at {service_url} — "
            f"NetworkPolicy may be blocking traffic"
        )
