"""Test suite for NeMo Guardrails MCP gateway integration."""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.nemo_guardrails import NemoGuardrails
from ocp_resources.route import Route

from tests.ai_safety.nemo_guardrails.constants import (
    CHAT_ENDPOINT,
    CHECK_ENDPOINT,
    MCP_ENVOY_FILTER_NAME,
    MCP_GATEWAY_NAME,
    MCP_GATEWAY_NAMESPACE,
    MODEL_NAME,
    SAFE_PROMPTS,
)
from tests.ai_safety.nemo_guardrails.utils import (
    send_request,
    wait_for_envoy_filter,
)
from utilities.certificates_utils import get_tls_verify


@pytest.mark.tier1
@pytest.mark.ai_safety
@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-nemo-guardrails"})],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed", "installed_istio", "installed_mcp_gateway", "bbr_envoy_filter")
class TestNemoGuardrailsMCP:
    """
    Tests for MCP gateway configuration in NeMo Guardrails.

    This test class validates:
    1. MCP gateway reference is stored in the CR spec
    2. Operator-created route exists and is reachable
    3. Operator creates an EnvoyFilter when MCPGatewayExtension is present
    """

    def test_nemo_mcp_cr_spec(
        self,
        nemo_guardrails_mcp: NemoGuardrails,
    ) -> None:
        """
        Verify MCP gateway is recorded in the CR spec.

        Given: NeMo Guardrails CR created with mcpGateway field
        When: CR spec is inspected
        Then: mcpGateway entry references the expected gateway name and namespace
        """
        mcp_gateway = nemo_guardrails_mcp.kind_dict["spec"]["mcpGateway"]
        assert mcp_gateway, "CR spec.mcpGateway should not be empty"
        assert mcp_gateway["name"] == MCP_GATEWAY_NAME, (
            f"Expected mcpGateway.name={MCP_GATEWAY_NAME!r}, got {mcp_gateway['name']!r}"
        )
        assert mcp_gateway["namespace"] == MCP_GATEWAY_NAMESPACE, (
            f"Expected mcpGateway.namespace={MCP_GATEWAY_NAMESPACE!r}, got {mcp_gateway['namespace']!r}"
        )

    def test_nemo_mcp_deployment(
        self,
        nemo_guardrails_mcp: NemoGuardrails,
        nemo_guardrails_mcp_route: Route,
    ) -> None:
        """
        Verify NeMo Guardrails server with MCP gateway config deploys and exposes a route.

        Given: NeMo Guardrails CR with MCP gateway config
        When: Operator reconciles the CR
        Then: CR exists and the operator-created route is reachable
        """
        assert nemo_guardrails_mcp.exists
        assert nemo_guardrails_mcp_route.exists
        assert nemo_guardrails_mcp_route.host is not None

    @pytest.mark.parametrize("endpoint", [CHAT_ENDPOINT, CHECK_ENDPOINT])
    def test_nemo_mcp_backend_communication(
        self,
        admin_client: DynamicClient,
        current_client_token: str,
        nemo_guardrails_mcp: NemoGuardrails,
        nemo_guardrails_mcp_route: Route,
        nemo_guardrails_mcp_healthcheck: None,
        endpoint: str,
    ) -> None:
        """
        Verify the NeMo Guardrails server with MCP gateway config can serve guardrail requests.

        Given: NeMo Guardrails with MCP gateway configured and auth enabled
        When: A valid authenticated request is sent
        Then: Server returns 200 with the expected response shape
        """
        url = f"https://{nemo_guardrails_mcp_route.host}{endpoint}"
        response = send_request(
            url=url,
            token=current_client_token,
            ca_bundle_file=get_tls_verify(client=admin_client),
            message=SAFE_PROMPTS[0],
            model=MODEL_NAME,
            configuration=None,
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        response_json = response.json()

        if endpoint == CHAT_ENDPOINT:
            assert "choices" in response_json, "Chat endpoint should return choices"
            assert len(response_json["choices"]) > 0
        elif endpoint == CHECK_ENDPOINT:
            assert "status" in response_json, "Check endpoint should return status"
            assert response_json["status"] in {"blocked", "success", "passed"}

    @pytest.mark.usefixtures("nemo_guardrails_mcp")
    def test_nemo_mcp_envoy_filter_created(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """
        Verify operator creates an EnvoyFilter once the MCPGatewayExtension is present.

        Given: An MCPGatewayExtension exists in the mcp-system namespace and NeMo Guardrails CR references it
        When: Operator reconciles the CR
        Then: An EnvoyFilter is created in the mcp-system namespace
        """
        envoy_filter = wait_for_envoy_filter(
            admin_client=admin_client,
            namespace=MCP_GATEWAY_NAMESPACE,
            name=MCP_ENVOY_FILTER_NAME,
        )

        assert envoy_filter.instance.spec.configPatches, (
            f"EnvoyFilter {MCP_ENVOY_FILTER_NAME!r} should contain configPatches"
        )
