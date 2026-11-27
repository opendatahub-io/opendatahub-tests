import pytest
from kubernetes.dynamic import DynamicClient

from tests.ai_gateway.models_as_a_service.multitenancy.aitenant.utils import tenant_namespace_name_from_aitenant
from tests.ai_gateway.models_as_a_service.multitenancy.utils import verify_maas_api_deployment_for_aitenant
from tests.ai_gateway.models_as_a_service.praxis.utils import (
    gateway_namespace_and_name_for_aitenant,
    migrate_legacy_aitenant_to_praxis_payload_processing,
    restore_legacy_aitenant_payload_processing,
    verify_legacy_ipp_installed_for_aitenant,
    verify_praxis_maas_tenant_config_ready,
    verify_praxis_payload_processing_active_for_aitenant,
)
from utilities.resources.aitenant import AITenant


@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest", "aitenant_infra_namespace")
class TestAITenantPraxisLegacyIpp:
    """Verify maas-controller legacy IPP opt-in vs default Praxis for AITenants."""

    @pytest.mark.tier1
    def test_praxis_aitenant_does_not_install_ipp_in_gateway_ns(
        self,
        admin_client: DynamicClient,
        ready_aitenant_default_dataplane: AITenant,
    ) -> None:
        """Given default Praxis on MaasTenantConfig, when controllers reconcile,
        then MaaS skips legacy IPP and ai-gateway installs the Praxis extproc bundle.
        """
        verify_praxis_payload_processing_active_for_aitenant(
            admin_client=admin_client,
            aitenant=ready_aitenant_default_dataplane,
        )

    @pytest.mark.tier1
    def test_legacy_ipp_to_praxis_removes_legacy_ipp_from_gateway_ns(
        self,
        admin_client: DynamicClient,
        ready_aitenant_legacy_ipp: AITenant,
    ) -> None:
        """Given legacy IPP on MaasTenantConfig, when switching to default Praxis,
        then MaaS releases legacy IPP and ai-gateway installs the Praxis bundle.
        """
        migrate_legacy_aitenant_to_praxis_payload_processing(
            admin_client=admin_client,
            aitenant=ready_aitenant_legacy_ipp,
        )

    @pytest.mark.smoke
    def test_praxis_aitenant_still_deploys_maas_api(
        self,
        admin_client: DynamicClient,
        ready_aitenant_default_dataplane: AITenant,
        maas_api_infra_namespace: str,
    ) -> None:
        """Given default Praxis, when platform reconciliation completes, then per-tenant maas-api is Available."""
        tenant_namespace_name = tenant_namespace_name_from_aitenant(aitenant=ready_aitenant_default_dataplane)
        verify_maas_api_deployment_for_aitenant(
            admin_client=admin_client,
            api_namespace=maas_api_infra_namespace,
            aitenant_name=ready_aitenant_default_dataplane.name,
            tenant_namespace_name=tenant_namespace_name,
        )

    @pytest.mark.smoke
    def test_praxis_aitenant_maas_tenant_config_ready(
        self,
        admin_client: DynamicClient,
        ready_aitenant_default_dataplane: AITenant,
    ) -> None:
        """Given default Praxis, when MaasTenantConfig reconciles,
        then default-tenant is Ready without legacy IPP EnvoyFilter dependency errors.
        """
        verify_praxis_maas_tenant_config_ready(
            admin_client=admin_client,
            aitenant=ready_aitenant_default_dataplane,
        )

    @pytest.mark.tier1
    def test_legacy_ipp_opt_in_installs_ipp(
        self,
        admin_client: DynamicClient,
        ready_aitenant_legacy_ipp: AITenant,
    ) -> None:
        """Given payload-processing-type=ipp on MaasTenantConfig, when bootstrap completes,
        then maas-controller installs legacy IPP in the gateway namespace.
        """
        gateway_namespace, _gateway_name = gateway_namespace_and_name_for_aitenant(
            aitenant=ready_aitenant_legacy_ipp,
        )
        verify_legacy_ipp_installed_for_aitenant(
            admin_client=admin_client,
            gateway_namespace=gateway_namespace,
            aitenant_name=ready_aitenant_legacy_ipp.name,
        )

    @pytest.mark.tier2
    def test_ipp_annotation_restores_legacy_ipp_path(
        self,
        admin_client: DynamicClient,
        ready_aitenant_default_dataplane: AITenant,
    ) -> None:
        """Given a tenant on default Praxis, when ipp is set on MaasTenantConfig,
        then maas-controller manages legacy IPP again in the gateway namespace.
        """
        restore_legacy_aitenant_payload_processing(
            admin_client=admin_client,
            aitenant=ready_aitenant_default_dataplane,
        )
