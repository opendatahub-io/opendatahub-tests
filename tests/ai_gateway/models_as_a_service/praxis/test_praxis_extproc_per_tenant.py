"""Per-tenant Praxis extproc workload tests (RHOAIENG-94553+)."""

import pytest
from kubernetes.dynamic import DynamicClient

from tests.ai_gateway.models_as_a_service.praxis.utils import (
    verify_aitenant_lacks_payload_processing_type_annotation,
    verify_maastenantconfig_has_praxis_cleanup_finalizer,
    verify_maastenantconfig_payload_processing_type,
    verify_praxis_payload_processing_active_for_aitenant,
)
from utilities.resources.aitenant import AITenant


@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest", "aitenant_infra_namespace")
class TestPraxisExtprocPerTenantWorkloads:
    """Verify per-tenant Praxis extproc Deployments for born-Praxis AITenants (RHOAIENG-94553)."""

    @pytest.mark.smoke
    def test_praxis_extproc_workloads_default_dataplane(
        self,
        admin_client: DynamicClient,
        ready_aitenant_bootstrap_for_default_dataplane: AITenant,
    ) -> None:
        """Given MaasTenantConfig without ipp (default Praxis), when bootstrap completes,
        then per-tenant payload-processing and payload-pre-processing Deployments exist
        in status.gatewayRef.namespace and reach Available with Praxis extproc configuration.
        """
        aitenant = ready_aitenant_bootstrap_for_default_dataplane
        verify_maastenantconfig_payload_processing_type(
            admin_client=admin_client,
            aitenant=aitenant,
            expected_value=None,
        )
        verify_maastenantconfig_has_praxis_cleanup_finalizer(admin_client=admin_client, aitenant=aitenant)
        verify_praxis_payload_processing_active_for_aitenant(admin_client=admin_client, aitenant=aitenant)
        verify_aitenant_lacks_payload_processing_type_annotation(aitenant=aitenant)
