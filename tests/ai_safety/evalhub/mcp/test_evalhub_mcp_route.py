import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service

from tests.ai_safety.evalhub.mcp.constants import (
    EVALHUB_MCP_CONFIGMAP_SUFFIX,
    EVALHUB_MCP_CR_NAME,
    EVALHUB_MCP_DEPLOYMENT_SUFFIX,
    EVALHUB_MCP_SERVICE_PORT,
)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-mcp-route"},
        ),
    ],
    indirect=True,
)
@pytest.mark.smoke
@pytest.mark.ai_safety
class TestEvalHubMcpRoute:
    """Operator-provisioned MCP Service, Route, ConfigMap, and auth Secret."""

    def test_evalhub_mcp_service_exposes_https_port(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_mcp_mt_deployment,
    ) -> None:
        """
        Given: EvalHub operator has provisioned the MCP service
        When: MCP Service ports are inspected
        Then: Service exposes port 8443 for kube-rbac-proxy
        """
        service_name = f"{EVALHUB_MCP_CR_NAME}{EVALHUB_MCP_DEPLOYMENT_SUFFIX}"
        service = Service(
            client=admin_client,
            name=service_name,
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        ports = service.instance.spec.ports
        assert ports, "Expected MCP service ports"
        assert ports[0].port == EVALHUB_MCP_SERVICE_PORT

    def test_evalhub_mcp_route_targets_service(
        self,
        evalhub_mcp_mt_route: Route,
        model_namespace: Namespace,
    ) -> None:
        """
        Given: EvalHub operator has provisioned the MCP route
        When: MCP Route spec is inspected
        Then: Route host is configured and targets the MCP service
        """
        assert evalhub_mcp_mt_route.host, "Expected non-empty MCP route host"
        assert evalhub_mcp_mt_route.namespace == model_namespace.name
        assert evalhub_mcp_mt_route.instance.spec.to.name == f"{EVALHUB_MCP_CR_NAME}{EVALHUB_MCP_DEPLOYMENT_SUFFIX}"

    def test_evalhub_mcp_configmap_exists(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_mcp_mt_cr,
    ) -> None:
        """
        Given: EvalHub CR with MCP enabled
        When: Operator-provisioned MCP ConfigMap is inspected
        Then: ConfigMap exists with config.yaml data
        """
        configmap_name = f"{EVALHUB_MCP_CR_NAME}{EVALHUB_MCP_CONFIGMAP_SUFFIX}"
        configmap = ConfigMap(
            client=admin_client,
            name=configmap_name,
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        data = configmap.instance.data or {}
        assert "config.yaml" in data, f"Expected config.yaml in {configmap_name}"

    def test_evalhub_mcp_auth_secret_exists(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_mcp_mt_cr,
    ) -> None:
        """
        Given: EvalHub CR with MCP enabled
        When: Operator-provisioned MCP auth Secret is inspected
        Then: Secret exists for outbound EvalHub API access
        """
        secret_name = f"{EVALHUB_MCP_CR_NAME}{EVALHUB_MCP_DEPLOYMENT_SUFFIX}-token"
        secret = Secret(
            client=admin_client,
            name=secret_name,
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        assert secret.exists
