import json

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.network_policy import NetworkPolicy

from tests.ai_hub.constants import (
    AUTOML_BFF_PEER,
    CURL_CONNECT_TIMEOUT_EXIT_CODE,
    KUBE_RBAC_PROXY_PORT,
    MODEL_REGISTRY_BASE_URI,
    ROUTER_INGRESS_PEER,
)
from tests.ai_hub.utils import (
    curl_from_pod,
    get_mr_service_by_label,
    get_network_policy_ingress_rules,
    get_service_external_address_rest,
)
from utilities.resources.model_registry_modelregistry_opendatahub_io import ModelRegistry
from utilities.resources.pod import Pod


@pytest.mark.parametrize(
    "model_registry_metadata_db_resources, model_registry_instance",
    [pytest.param({}, {}, id="test_mr_network_policy")],
    indirect=True,
)
@pytest.mark.usefixtures("updated_dsc_component_state_scope_session")
class TestModelRegistryNetworkPolicy:
    """Validate that the model registry https-route NetworkPolicy allows only router/gateway and AutoML BFF traffic."""

    def test_https_route_ingress_namespace_labels(
        self,
        admin_client: DynamicClient,
        model_registry_instance: list[ModelRegistry],
        model_registry_namespace: str,
    ) -> None:
        """Given a deployed model registry instance
        When inspecting its https-route NetworkPolicy ingress rules
        Then it allows only the OpenShift router/gateway namespaces and AutoML BFF pods in the applications
        namespace, on the kube-rbac-proxy port only
        """
        np_name = f"{model_registry_instance[0].name}-https-route"
        network_policy = NetworkPolicy(
            client=admin_client,
            name=np_name,
            namespace=model_registry_namespace,
            ensure_exists=True,
        )
        ingress_rules = get_network_policy_ingress_rules(network_policy=network_policy)
        assert len(ingress_rules) == 1, f"{np_name} should have exactly one ingress rule: {ingress_rules}"
        # NetworkPolicy peer order has no meaning, so compare independent of order
        expected_peers = sorted([ROUTER_INGRESS_PEER, AUTOML_BFF_PEER], key=json.dumps)
        assert sorted(ingress_rules[0]["from"], key=json.dumps) == expected_peers, (
            f"{np_name} should allow traffic only from the OpenShift router/gateway namespaces and AutoML BFF pods, "
            f"not the whole applications namespace: {ingress_rules[0]['from']}"
        )
        assert ingress_rules[0]["ports"] == [{"port": KUBE_RBAC_PROXY_PORT, "protocol": "TCP"}], (
            f"{np_name} should allow only TCP {KUBE_RBAC_PROXY_PORT}: {ingress_rules[0]['ports']}"
        )

    def test_dashboard_cannot_reach_model_registry_service_directly(
        self,
        dashboard_pod: Pod,
        model_registry_instance: list[ModelRegistry],
        model_registry_namespace: str,
    ) -> None:
        """Given a dashboard pod in the shared applications namespace
        When curling the model registry internal service on the kube-rbac-proxy port
        Then the connection times out because the NetworkPolicy blocks it
        """
        mr_name = model_registry_instance[0].name
        service_url = f"https://{mr_name}.{model_registry_namespace}.svc.cluster.local:{KUBE_RBAC_PROXY_PORT}"
        result = curl_from_pod(pod=dashboard_pod, url=service_url)
        assert result.name_lookup_seconds > 0, f"{service_url} should resolve from the dashboard pod: {result}"
        assert result.rc == CURL_CONNECT_TIMEOUT_EXIT_CODE and result.tcp_connect_seconds == 0, (
            f"Dashboard pod should be blocked from opening a TCP connection to {service_url}: {result}"
        )

    def test_dashboard_can_reach_model_registry_through_gateway(
        self,
        admin_client: DynamicClient,
        dashboard_pod: Pod,
        model_registry_instance: list[ModelRegistry],
        model_registry_namespace: str,
    ) -> None:
        """Given a dashboard pod in the shared applications namespace
        When curling the model registry external REST address that the Dashboard BFF uses
        Then the request succeeds end to end through the router/gateway
        """
        mr_service = get_mr_service_by_label(
            client=admin_client,
            namespace_name=model_registry_namespace,
            mr_instance=model_registry_instance[0],
        )
        external_address = get_service_external_address_rest(service=mr_service)
        assert external_address, f"{mr_service.name} Service should have an external-address-rest annotation"
        url = f"https://{external_address}{MODEL_REGISTRY_BASE_URI}registered_models"
        result = curl_from_pod(pod=dashboard_pod, url=url, use_service_account_token=True)
        assert result.rc == 0 and result.http_code == 200, (
            f"Dashboard pod cannot reach model registry at {url}: {result}"
        )
