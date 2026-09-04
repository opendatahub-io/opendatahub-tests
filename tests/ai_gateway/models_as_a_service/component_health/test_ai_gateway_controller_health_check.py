import pytest
from kubernetes.dynamic import DynamicClient

from tests.ai_gateway.models_as_a_service.component_health.utils import (
    verify_ai_gateway_controller_deployment_available,
    verify_ai_gateway_controller_health_probes_configured,
    verify_ai_gateway_controller_parameters_configmap,
    verify_ai_gateway_controller_pods_running,
    verify_ai_gateway_controller_praxis_image_env_from_configmap,
    verify_ai_gateway_controller_rbac_exists,
    verify_aigateway_models_as_a_service_ready,
)


@pytest.mark.component_health
@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest")
class TestAIGatewayControllerHealth:
    def test_ai_gateway_controller_deployment_available(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Given MaaS is managed, when checking applications namespace, then ai-gateway-controller is Available."""
        verify_ai_gateway_controller_deployment_available(admin_client=admin_client)

    def test_ai_gateway_controller_pods_running(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Given ai-gateway-controller is deployed, when checking its pods, then they are Running and ready."""
        verify_ai_gateway_controller_pods_running(admin_client=admin_client)

    def test_ai_gateway_controller_rbac_exists(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Given MaaS is managed, when checking cluster RBAC, then ai-gateway-controller RBAC is present."""
        verify_ai_gateway_controller_rbac_exists(admin_client=admin_client)

    def test_ai_gateway_controller_parameters_configmap(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Given ai-gateway-controller is deployed, when reading its parameters ConfigMap, then image keys exist."""
        verify_ai_gateway_controller_parameters_configmap(admin_client=admin_client)

    def test_aigateway_cr_models_as_a_service_ready(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Given AIGateway CR exists, when checking status, then ModelsAsAServiceReady is True."""
        verify_aigateway_models_as_a_service_ready(admin_client=admin_client)

    def test_ai_gateway_controller_health_probes_configured(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Given ai-gateway-controller Deployment exists, when inspecting manager container, then probes are set."""
        verify_ai_gateway_controller_health_probes_configured(admin_client=admin_client)

    def test_ai_gateway_controller_praxis_image_env_from_configmap(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Given ai-gateway-controller is deployed, when checking env wiring, then praxis image comes from ConfigMap."""
        verify_ai_gateway_controller_praxis_image_env_from_configmap(admin_client=admin_client)
