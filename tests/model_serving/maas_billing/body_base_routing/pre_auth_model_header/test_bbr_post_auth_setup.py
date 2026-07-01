"""Smoke tests for BBR payload-processing (post-auth) infrastructure deployment."""

from typing import Self

import pytest
from kubernetes.dynamic import DynamicClient

from tests.model_serving.maas_billing.body_base_routing.pre_auth_model_header.utils import (
    verify_bbr_plugins_configmap_has_expected_plugins,
    verify_bbr_post_auth_processing_deployment_ready,
    verify_bbr_post_processing_destination_rule_exists,
    verify_bbr_post_processing_service_port,
)


@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest")
class TestBBRPostAuthSetup:
    """Smoke tests for the BBR post-auth ext_proc stack (Deployment, Service, DestinationRule)."""

    @pytest.mark.smoke
    def test_bbr_post_processing_deployment_ready(
        self: Self,
        admin_client: DynamicClient,
        bbr_gateway_namespace: str,
    ) -> None:
        """Verify the payload-processing Deployment exists with all replicas ready."""
        verify_bbr_post_auth_processing_deployment_ready(
            admin_client=admin_client,
            gateway_namespace=bbr_gateway_namespace,
        )

    @pytest.mark.smoke
    def test_bbr_post_processing_service_exposes_port_9004(
        self: Self,
        admin_client: DynamicClient,
        bbr_gateway_namespace: str,
    ) -> None:
        """Verify the payload-processing Service exposes port 9004 for the bbr post-auth ext_proc gRPC connection."""
        verify_bbr_post_processing_service_port(
            admin_client=admin_client,
            gateway_namespace=bbr_gateway_namespace,
        )

    @pytest.mark.smoke
    def test_bbr_post_processing_destination_rule_exists(
        self: Self,
        admin_client: DynamicClient,
        bbr_gateway_namespace: str,
    ) -> None:
        """Verify an Istio DestinationRule exists in the gateway namespace for the post-auth processing pod."""
        verify_bbr_post_processing_destination_rule_exists(
            admin_client=admin_client,
            gateway_namespace=bbr_gateway_namespace,
        )

    @pytest.mark.smoke
    def test_bbr_plugins_configmap_has_expected_plugins(
        self: Self,
        admin_client: DynamicClient,
        bbr_gateway_namespace: str,
    ) -> None:
        """Verify the payload-processing-plugins ConfigMap contains the expected post-auth and pre-auth plugin types."""
        verify_bbr_plugins_configmap_has_expected_plugins(
            admin_client=admin_client,
            gateway_namespace=bbr_gateway_namespace,
        )
