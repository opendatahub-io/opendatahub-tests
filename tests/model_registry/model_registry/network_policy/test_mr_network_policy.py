import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.network_policy import NetworkPolicy

from utilities.resources.model_registry_modelregistry_opendatahub_io import ModelRegistry


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
