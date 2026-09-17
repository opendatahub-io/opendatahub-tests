import pytest
from kubernetes.dynamic import DynamicClient

from tests.ai_gateway.models_as_a_service.multitenancy.aitenant.utils import tenant_namespace_name_from_aitenant
from tests.ai_gateway.models_as_a_service.multitenancy.utils import verify_maas_api_deployment_for_aitenant
from tests.ai_gateway.models_as_a_service.praxis.constants import (
    LEGACY_IPP_SWITCH_BACK_WAIT_TIMEOUT_SECONDS,
    PRAXIS_PAYLOAD_PROCESSING_TYPE_VALUE,
)
from tests.ai_gateway.models_as_a_service.praxis.utils import (
    gateway_namespace_and_name_for_aitenant,
    set_aitenant_payload_processing_type_annotation,
    verify_aitenant_has_praxis_cleanup_finalizer,
    verify_legacy_ipp_installed_for_aitenant,
    verify_legacy_ipp_not_installed_for_aitenant,
    verify_praxis_maas_tenant_config_ready,
    wait_until_aitenant_lacks_praxis_cleanup_finalizer,
)
from utilities.resources.aitenant import AITenant


@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest", "aitenant_infra_namespace")
class TestAITenantPraxisLegacyIpp:
    """Verify maas-controller legacy IPP behavior for praxis vs legacy AITenants."""

    @pytest.mark.tier1
    def test_praxis_aitenant_does_not_install_ipp_in_gateway_ns(
        self,
        admin_client: DynamicClient,
        ready_praxis_annotated_aitenant: AITenant,
    ) -> None:
        """Given a Ready praxis-annotated AITenant, when maas-controller reconciles,
        then maas legacy IPP markers (custom-ipp-config.yaml) are not present in the gateway namespace.
        """
        gateway_namespace, _gateway_name = gateway_namespace_and_name_for_aitenant(
            aitenant=ready_praxis_annotated_aitenant,
        )
        verify_legacy_ipp_not_installed_for_aitenant(
            admin_client=admin_client,
            gateway_namespace=gateway_namespace,
            aitenant_name=ready_praxis_annotated_aitenant.name,
        )

    @pytest.mark.tier1
    def test_praxis_annotation_removes_legacy_ipp_from_gateway_ns(
        self,
        admin_client: DynamicClient,
        ready_aitenant_without_praxis_annotation: AITenant,
    ) -> None:
        """Given a legacy AITenant with maas legacy IPP installed, when the praxis annotation is applied,
        then maas legacy IPP markers are removed from the gateway namespace.
        """
        gateway_namespace, _gateway_name = gateway_namespace_and_name_for_aitenant(
            aitenant=ready_aitenant_without_praxis_annotation,
        )
        aitenant_name = ready_aitenant_without_praxis_annotation.name
        verify_legacy_ipp_installed_for_aitenant(
            admin_client=admin_client,
            gateway_namespace=gateway_namespace,
            aitenant_name=aitenant_name,
        )
        set_aitenant_payload_processing_type_annotation(
            aitenant=ready_aitenant_without_praxis_annotation,
            annotation_value=PRAXIS_PAYLOAD_PROCESSING_TYPE_VALUE,
        )
        verify_aitenant_has_praxis_cleanup_finalizer(aitenant=ready_aitenant_without_praxis_annotation)
        verify_legacy_ipp_not_installed_for_aitenant(
            admin_client=admin_client,
            gateway_namespace=gateway_namespace,
            aitenant_name=aitenant_name,
        )

    @pytest.mark.smoke
    def test_praxis_aitenant_still_deploys_maas_api(
        self,
        admin_client: DynamicClient,
        ready_praxis_annotated_aitenant: AITenant,
        maas_api_infra_namespace: str,
    ) -> None:
        """Given a Ready praxis-annotated AITenant, when platform reconciliation completes,
        then per-tenant maas-api is still Available.
        """
        tenant_namespace_name = tenant_namespace_name_from_aitenant(aitenant=ready_praxis_annotated_aitenant)
        verify_maas_api_deployment_for_aitenant(
            admin_client=admin_client,
            api_namespace=maas_api_infra_namespace,
            aitenant_name=ready_praxis_annotated_aitenant.name,
            tenant_namespace_name=tenant_namespace_name,
        )

    @pytest.mark.smoke
    def test_praxis_aitenant_maas_tenant_config_ready(
        self,
        admin_client: DynamicClient,
        ready_praxis_annotated_aitenant: AITenant,
    ) -> None:
        """Given a Ready praxis-annotated AITenant, when MaasTenantConfig reconciles,
        then default-tenant is Ready without legacy IPP EnvoyFilter dependency errors.
        """
        verify_praxis_maas_tenant_config_ready(
            admin_client=admin_client,
            aitenant=ready_praxis_annotated_aitenant,
        )

    @pytest.mark.tier1
    def test_legacy_aitenant_still_installs_ipp(
        self,
        admin_client: DynamicClient,
        ready_aitenant_without_praxis_annotation: AITenant,
    ) -> None:
        """Given an unannotated legacy AITenant, when bootstrap completes,
        then maas-controller installs legacy IPP in the gateway namespace.
        """
        gateway_namespace, _gateway_name = gateway_namespace_and_name_for_aitenant(
            aitenant=ready_aitenant_without_praxis_annotation,
        )
        verify_legacy_ipp_installed_for_aitenant(
            admin_client=admin_client,
            gateway_namespace=gateway_namespace,
            aitenant_name=ready_aitenant_without_praxis_annotation.name,
        )

    @pytest.mark.tier2
    def test_deannotate_restores_legacy_ipp_path(
        self,
        admin_client: DynamicClient,
        ready_aitenant_without_praxis_annotation: AITenant,
    ) -> None:
        """Given a tenant switched to praxis and back to legacy, when the annotation is removed,
        then maas-controller manages legacy IPP again in the gateway namespace.
        """
        gateway_namespace, _gateway_name = gateway_namespace_and_name_for_aitenant(
            aitenant=ready_aitenant_without_praxis_annotation,
        )
        aitenant_name = ready_aitenant_without_praxis_annotation.name
        verify_legacy_ipp_installed_for_aitenant(
            admin_client=admin_client,
            gateway_namespace=gateway_namespace,
            aitenant_name=aitenant_name,
        )
        set_aitenant_payload_processing_type_annotation(
            aitenant=ready_aitenant_without_praxis_annotation,
            annotation_value=PRAXIS_PAYLOAD_PROCESSING_TYPE_VALUE,
        )
        verify_aitenant_has_praxis_cleanup_finalizer(aitenant=ready_aitenant_without_praxis_annotation)
        verify_legacy_ipp_not_installed_for_aitenant(
            admin_client=admin_client,
            gateway_namespace=gateway_namespace,
            aitenant_name=aitenant_name,
        )
        set_aitenant_payload_processing_type_annotation(
            aitenant=ready_aitenant_without_praxis_annotation,
            annotation_value=None,
        )
        wait_until_aitenant_lacks_praxis_cleanup_finalizer(
            aitenant=ready_aitenant_without_praxis_annotation,
            timeout=LEGACY_IPP_SWITCH_BACK_WAIT_TIMEOUT_SECONDS,
        )
        verify_legacy_ipp_installed_for_aitenant(
            admin_client=admin_client,
            gateway_namespace=gateway_namespace,
            aitenant_name=aitenant_name,
            timeout=LEGACY_IPP_SWITCH_BACK_WAIT_TIMEOUT_SECONDS,
        )
