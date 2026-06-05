"""Smoke tests for BBR payload-pre-processing infrastructure deployment."""

import pytest
from kubernetes.dynamic import DynamicClient

from tests.model_serving.maas_billing.body_base_routing.pre_auth_model_header.utils import (
    verify_bbr_pre_processing_deployment_ready,
    verify_bbr_pre_processing_destination_rule_exists,
    verify_bbr_pre_processing_service_port,
)


@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest")
class TestBBRSetup:
    """Smoke tests for the BBR pre-auth ext_proc stack (Deployment, Service, DestinationRule)."""

    @pytest.mark.smoke
    def test_bbr_pre_processing_deployment_ready(
        self,
        admin_client: DynamicClient,
        bbr_gateway_namespace: str,
    ) -> None:
        """Verify the payload-pre-processing Deployment exists with all replicas ready."""
        verify_bbr_pre_processing_deployment_ready(
            admin_client=admin_client,
            gateway_namespace=bbr_gateway_namespace,
        )

    @pytest.mark.smoke
    def test_bbr_pre_processing_service_exposes_port_9004(
        self,
        admin_client: DynamicClient,
        bbr_gateway_namespace: str,
    ) -> None:
        """Verify the payload-pre-processing Service exposes port 9004 for the bbr-pre ext_proc gRPC connection."""
        verify_bbr_pre_processing_service_port(
            admin_client=admin_client,
            gateway_namespace=bbr_gateway_namespace,
        )

    @pytest.mark.smoke
    def test_bbr_pre_processing_destination_rule_exists(
        self,
        admin_client: DynamicClient,
        bbr_gateway_namespace: str,
    ) -> None:
        """Verify an Istio DestinationRule exists in the gateway namespace for the pre-processing pod."""
        verify_bbr_pre_processing_destination_rule_exists(
            admin_client=admin_client,
            gateway_namespace=bbr_gateway_namespace,
        )
