import pytest
from kubernetes.dynamic import DynamicClient

from tests.ai_gateway.models_as_a_service.praxis.constants import PRAXIS_PAYLOAD_PROCESSING_TYPE_VALUE
from tests.ai_gateway.models_as_a_service.praxis.utils import (
    deploy_praxis_aitenant_and_verify_maastenantconfig_annotation,
    praxis_aitenant_with_bootstrap_gateway,
    verify_aitenant_bootstrap_reaches_ready_with_refs,
    verify_aitenant_lacks_payload_processing_type_annotation,
    verify_maastenantconfig_has_praxis_cleanup_finalizer,
    verify_maastenantconfig_lacks_praxis_cleanup_finalizer,
    verify_maastenantconfig_non_praxis_payload_processing_uses_legacy_ipp,
    verify_maastenantconfig_payload_processing_type,
)
from tests.ai_gateway.models_as_a_service.utils import deploy_and_verify_aitenant_ready
from utilities.resources.aitenant import AITenant


@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest", "aitenant_infra_namespace")
class TestMaasTenantConfigPraxisAnnotation:
    """Verify MaasTenantConfig payload-processing-type contract for Praxis opt-in."""

    @pytest.mark.tier1
    def test_maastenantconfig_accepts_praxis_annotation(
        self,
        admin_client: DynamicClient,
        aitenant_infra_namespace: str,
        teardown_resources: bool,
    ) -> None:
        """Given a Ready AITenant, when praxis is set on MaasTenantConfig,
        then the annotation is persisted and the AITenant does not mirror it.
        """
        with praxis_aitenant_with_bootstrap_gateway(
            admin_client=admin_client,
            cr_namespace=aitenant_infra_namespace,
            teardown=teardown_resources,
        ) as aitenant:
            deploy_praxis_aitenant_and_verify_maastenantconfig_annotation(
                admin_client=admin_client,
                aitenant=aitenant,
                expected_annotation_value=PRAXIS_PAYLOAD_PROCESSING_TYPE_VALUE,
            )
            verify_maastenantconfig_has_praxis_cleanup_finalizer(admin_client=admin_client, aitenant=aitenant)
            verify_aitenant_lacks_payload_processing_type_annotation(aitenant=aitenant)

    @pytest.mark.smoke
    def test_praxis_opt_in_aitenant_bootstrap_reaches_ready(
        self,
        admin_client: DynamicClient,
        ready_praxis_annotated_aitenant: AITenant,
    ) -> None:
        """Given MaasTenantConfig opted into praxis, when bootstrap completes,
        then the AITenant is Ready with status refs populated.
        """
        verify_aitenant_bootstrap_reaches_ready_with_refs(aitenant=ready_praxis_annotated_aitenant)
        verify_maastenantconfig_has_praxis_cleanup_finalizer(
            admin_client=admin_client,
            aitenant=ready_praxis_annotated_aitenant,
        )

    @pytest.mark.tier1
    def test_maastenantconfig_without_annotation_defaults_to_legacy_ipp(
        self,
        admin_client: DynamicClient,
        ready_aitenant_without_praxis_annotation: AITenant,
    ) -> None:
        """Given MaasTenantConfig without payload-processing-type, when bootstrap completes,
        then the annotation is absent, Praxis cleanup is not enabled, and the AITenant is not annotated.
        """
        verify_maastenantconfig_payload_processing_type(
            admin_client=admin_client,
            aitenant=ready_aitenant_without_praxis_annotation,
            expected_value=None,
        )
        verify_maastenantconfig_lacks_praxis_cleanup_finalizer(
            admin_client=admin_client,
            aitenant=ready_aitenant_without_praxis_annotation,
        )
        verify_aitenant_lacks_payload_processing_type_annotation(
            aitenant=ready_aitenant_without_praxis_annotation,
        )

    @pytest.mark.tier2
    @pytest.mark.parametrize(
        "annotation_value",
        [
            pytest.param("foo", id="test_non_praxis_value"),
            pytest.param("", id="test_empty_value"),
            pytest.param("ipp", id="test_ipp_value"),
        ],
    )
    def test_maastenantconfig_non_praxis_payload_processing_uses_legacy_ipp(
        self,
        admin_client: DynamicClient,
        aitenant_infra_namespace: str,
        teardown_resources: bool,
        annotation_value: str,
    ) -> None:
        """Given a non-praxis payload-processing-type on MaasTenantConfig, when bootstrap completes,
        then the value is stored but Praxis cleanup is not enabled.
        """
        with praxis_aitenant_with_bootstrap_gateway(
            admin_client=admin_client,
            cr_namespace=aitenant_infra_namespace,
            teardown=teardown_resources,
        ) as aitenant:
            deploy_and_verify_aitenant_ready(aitenant=aitenant)
            verify_maastenantconfig_non_praxis_payload_processing_uses_legacy_ipp(
                admin_client=admin_client,
                aitenant=aitenant,
                annotation_value=annotation_value,
            )
