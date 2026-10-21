"""Per-tenant Praxis extproc workload tests (RHOAIENG-94553+)."""

import pytest
from kubernetes.dynamic import DynamicClient

from tests.ai_gateway.models_as_a_service.praxis.utils import (
    praxis_extproc_deployment_names_for_aitenant,
    read_maastenantconfig_payload_processing_type,
    verify_maastenantconfig_payload_processing_type,
    verify_praxis_extproc_deployment_available,
    verify_praxis_extproc_deployment_pods_ready,
    verify_praxis_extproc_deployments_exist_in_gateway_namespace,
    verify_praxis_payload_processing_active_for_aitenant,
    verify_praxis_workloads_installed_in_aitenant_gateway_namespace,
)
from utilities.resources.aitenant import AITenant


@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest", "aitenant_infra_namespace")
class TestPraxisExtprocPerTenantWorkloads:
    """Verify per-tenant Praxis extproc Deployments for born-Praxis AITenants (RHOAIENG-94553)."""

    @pytest.mark.tier1
    def test_praxis_resources_installed_default_dataplane(
        self,
        admin_client: DynamicClient,
        ready_aitenant_default_dataplane: AITenant,
    ) -> None:
        """Given MaasTenantConfig without ipp (default Praxis), when bootstrap completes,
        then Praxis extproc Deployments exist and are Available in the gateway namespace.
        """
        aitenant = ready_aitenant_default_dataplane
        verify_maastenantconfig_payload_processing_type(
            admin_client=admin_client,
            aitenant=aitenant,
            expected_value=None,
        )
        verify_praxis_payload_processing_active_for_aitenant(admin_client=admin_client, aitenant=aitenant)
        verify_praxis_extproc_deployments_exist_in_gateway_namespace(admin_client=admin_client, aitenant=aitenant)

    @pytest.mark.tier1
    def test_praxis_resources_installed_in_gateway_namespace(
        self,
        admin_client: DynamicClient,
        ready_aitenant_default_dataplane: AITenant,
    ) -> None:
        """Given a Praxis-ready AITenant, when reconciliation completes,
        then extproc workloads are installed under status.gatewayRef.namespace.
        """
        verify_praxis_workloads_installed_in_aitenant_gateway_namespace(
            admin_client=admin_client,
            aitenant=ready_aitenant_default_dataplane,
        )

    @pytest.mark.tier1
    def test_praxis_payload_processing_deployment_available(
        self,
        admin_client: DynamicClient,
        ready_aitenant_default_dataplane: AITenant,
    ) -> None:
        """Given default Praxis dataplane, then payload-processing Deployment is Available in gatewayRef.namespace."""
        gateway_namespace, post_processing_name, _pre_processing_name = praxis_extproc_deployment_names_for_aitenant(
            aitenant=ready_aitenant_default_dataplane,
        )
        verify_praxis_extproc_deployment_available(
            admin_client=admin_client,
            gateway_namespace=gateway_namespace,
            deployment_name=post_processing_name,
        )

    @pytest.mark.tier1
    def test_praxis_payload_pre_processing_deployment_available(
        self,
        admin_client: DynamicClient,
        ready_aitenant_default_dataplane: AITenant,
    ) -> None:
        """Given default Praxis dataplane, then payload-pre-processing Deployment is Available."""
        gateway_namespace, _post_processing_name, pre_processing_name = praxis_extproc_deployment_names_for_aitenant(
            aitenant=ready_aitenant_default_dataplane,
        )
        verify_praxis_extproc_deployment_available(
            admin_client=admin_client,
            gateway_namespace=gateway_namespace,
            deployment_name=pre_processing_name,
        )

    @pytest.mark.smoke
    def test_praxis_payload_processing_pods_ready(
        self,
        admin_client: DynamicClient,
        ready_aitenant_default_dataplane: AITenant,
    ) -> None:
        """Given Praxis extproc Deployments, then all replicas report ready for post- and pre-processing."""
        gateway_namespace, post_processing_name, pre_processing_name = praxis_extproc_deployment_names_for_aitenant(
            aitenant=ready_aitenant_default_dataplane,
        )
        payload_processing_type = read_maastenantconfig_payload_processing_type(
            admin_client=admin_client,
            aitenant=ready_aitenant_default_dataplane,
        )
        assert payload_processing_type is None, (
            f"Expected default Praxis (no ipp annotation) for smoke workload check, got {payload_processing_type!r}"
        )
        for deployment_name in (post_processing_name, pre_processing_name):
            verify_praxis_extproc_deployment_pods_ready(
                admin_client=admin_client,
                gateway_namespace=gateway_namespace,
                deployment_name=deployment_name,
            )
