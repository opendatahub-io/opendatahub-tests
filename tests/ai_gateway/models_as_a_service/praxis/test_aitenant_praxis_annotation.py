import pytest
from kubernetes.dynamic import DynamicClient

from tests.ai_gateway.models_as_a_service.praxis.constants import PRAXIS_PAYLOAD_PROCESSING_TYPE_VALUE
from tests.ai_gateway.models_as_a_service.praxis.utils import (
    opt_in_legacy_ipp_payload_processing_for_aitenant,
    praxis_aitenant_with_bootstrap_gateway,
    set_maastenantconfig_payload_processing_type_annotation,
    verify_aitenant_bootstrap_reaches_ready_with_refs,
    verify_aitenant_lacks_payload_processing_type_annotation,
    verify_maastenantconfig_has_praxis_cleanup_finalizer,
    verify_maastenantconfig_payload_processing_type,
    verify_praxis_payload_processing_active_for_aitenant,
    verify_unrecognized_payload_processing_type_uses_praxis_dataplane,
)
from tests.ai_gateway.models_as_a_service.utils import deploy_and_verify_aitenant_ready
from utilities.resources.aitenant import AITenant


@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest", "aitenant_infra_namespace")
class TestMaasTenantConfigPraxisAnnotation:
    """Verify MaasTenantConfig payload-processing-type: default Praxis, legacy via ipp."""

    @pytest.mark.tier1
    def test_maastenantconfig_accepts_explicit_praxis_annotation(
        self,
        admin_client: DynamicClient,
        aitenant_infra_namespace: str,
        teardown_resources: bool,
    ) -> None:
        """Given a legacy-IPP Ready AITenant, when praxis is set on MaasTenantConfig,
        then the annotation is persisted and the AITenant does not mirror it.
        """
        with praxis_aitenant_with_bootstrap_gateway(
            admin_client=admin_client,
            cr_namespace=aitenant_infra_namespace,
            teardown=teardown_resources,
        ) as aitenant:
            deploy_and_verify_aitenant_ready(aitenant=aitenant)
            opt_in_legacy_ipp_payload_processing_for_aitenant(admin_client=admin_client, aitenant=aitenant)
            set_maastenantconfig_payload_processing_type_annotation(
                admin_client=admin_client,
                aitenant=aitenant,
                annotation_value=PRAXIS_PAYLOAD_PROCESSING_TYPE_VALUE,
            )
            verify_maastenantconfig_payload_processing_type(
                admin_client=admin_client,
                aitenant=aitenant,
                expected_value=PRAXIS_PAYLOAD_PROCESSING_TYPE_VALUE,
            )
            verify_maastenantconfig_has_praxis_cleanup_finalizer(admin_client=admin_client, aitenant=aitenant)
            verify_praxis_payload_processing_active_for_aitenant(admin_client=admin_client, aitenant=aitenant)
            verify_aitenant_lacks_payload_processing_type_annotation(aitenant=aitenant)

    @pytest.mark.smoke
    def test_default_praxis_aitenant_bootstrap_reaches_ready(
        self,
        admin_client: DynamicClient,
        ready_aitenant_default_dataplane: AITenant,
    ) -> None:
        """Given default Praxis on MaasTenantConfig, then the AITenant is Ready with status refs populated."""
        verify_aitenant_bootstrap_reaches_ready_with_refs(aitenant=ready_aitenant_default_dataplane)
        verify_maastenantconfig_has_praxis_cleanup_finalizer(
            admin_client=admin_client,
            aitenant=ready_aitenant_default_dataplane,
        )

    @pytest.mark.tier1
    def test_maastenantconfig_without_annotation_defaults_to_praxis(
        self,
        ready_aitenant_default_dataplane: AITenant,
    ) -> None:
        """Given MaasTenantConfig without payload-processing-type, when bootstrap completes,
        then controllers use default Praxis and the AITenant is not annotated.

        Assertions run in ``ready_aitenant_default_dataplane`` via
        ``verify_default_dataplane_praxis_for_aitenant``.
        """

    @pytest.mark.tier1
    def test_maastenantconfig_ipp_opt_in_uses_legacy_ipp(
        self,
        ready_aitenant_legacy_ipp: AITenant,
    ) -> None:
        """Given payload-processing-type=ipp on MaasTenantConfig, when bootstrap completes,
        then legacy IPP is active and Praxis cleanup is not enabled.

        Assertions run in ``ready_aitenant_legacy_ipp`` via
        ``opt_in_legacy_ipp_payload_processing_for_aitenant``.
        """

    @pytest.mark.tier2
    @pytest.mark.parametrize(
        "annotation_value",
        [
            pytest.param("foo", id="test_non_praxis_value"),
            pytest.param("", id="test_empty_value"),
        ],
    )
    def test_maastenantconfig_unrecognized_type_defaults_to_praxis(
        self,
        admin_client: DynamicClient,
        aitenant_infra_namespace: str,
        teardown_resources: bool,
        annotation_value: str,
    ) -> None:
        """Given an unrecognized payload-processing-type on MaasTenantConfig, when reconciliation completes,
        then controllers keep the Praxis dataplane.
        """
        with praxis_aitenant_with_bootstrap_gateway(
            admin_client=admin_client,
            cr_namespace=aitenant_infra_namespace,
            teardown=teardown_resources,
        ) as aitenant:
            deploy_and_verify_aitenant_ready(aitenant=aitenant)
            verify_unrecognized_payload_processing_type_uses_praxis_dataplane(
                admin_client=admin_client,
                aitenant=aitenant,
                annotation_value=annotation_value,
            )
