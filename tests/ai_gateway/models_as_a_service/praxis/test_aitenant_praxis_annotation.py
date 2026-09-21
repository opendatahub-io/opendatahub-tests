import pytest
from kubernetes.dynamic import DynamicClient

from tests.ai_gateway.models_as_a_service.praxis.constants import PRAXIS_PAYLOAD_PROCESSING_TYPE_VALUE
from tests.ai_gateway.models_as_a_service.praxis.utils import (
    deploy_praxis_aitenant_and_verify_annotation,
    praxis_aitenant_with_bootstrap_gateway,
    verify_aitenant_bootstrap_reaches_ready_with_refs,
    verify_aitenant_has_praxis_cleanup_finalizer,
    verify_aitenant_lacks_praxis_cleanup_finalizer,
    verify_aitenant_payload_processing_annotation,
)
from tests.ai_gateway.models_as_a_service.utils import deploy_and_verify_aitenant_ready
from utilities.resources.aitenant import AITenant


@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest", "aitenant_infra_namespace")
class TestAITenantPraxisAnnotation:
    """Verify the AITenant payload-processing-type annotation contract for Praxis opt-in."""

    @pytest.mark.tier1
    def test_aitenant_accepts_praxis_annotation(
        self,
        admin_client: DynamicClient,
        aitenant_infra_namespace: str,
        teardown_resources: bool,
    ) -> None:
        """Given a bootstrap Gateway, when an AITenant is created with payload-processing-type praxis,
        then the annotation is persisted on the AITenant.
        """
        with praxis_aitenant_with_bootstrap_gateway(
            admin_client=admin_client,
            cr_namespace=aitenant_infra_namespace,
            payload_processing_type=PRAXIS_PAYLOAD_PROCESSING_TYPE_VALUE,
            teardown=teardown_resources,
        ) as aitenant:
            deploy_praxis_aitenant_and_verify_annotation(
                aitenant=aitenant,
                expected_annotation_value=PRAXIS_PAYLOAD_PROCESSING_TYPE_VALUE,
            )
            deploy_and_verify_aitenant_ready(aitenant=aitenant)
            verify_aitenant_has_praxis_cleanup_finalizer(aitenant=aitenant)

    @pytest.mark.smoke
    def test_annotated_aitenant_bootstrap_reaches_ready(
        self,
        ready_praxis_annotated_aitenant: AITenant,
    ) -> None:
        """Given an AITenant annotated for praxis, when bootstrap completes,
        then the AITenant is Ready and status refs are populated.
        """
        verify_aitenant_bootstrap_reaches_ready_with_refs(aitenant=ready_praxis_annotated_aitenant)
        verify_aitenant_has_praxis_cleanup_finalizer(aitenant=ready_praxis_annotated_aitenant)

    @pytest.mark.tier1
    def test_aitenant_without_annotation_defaults_to_legacy_ipp(
        self,
        ready_aitenant_without_praxis_annotation: AITenant,
    ) -> None:
        """Given an AITenant without payload-processing-type, when bootstrap completes,
        then the annotation is absent and the Praxis cleanup finalizer is not attached.
        """
        verify_aitenant_payload_processing_annotation(
            aitenant=ready_aitenant_without_praxis_annotation, expected_value=None
        )
        verify_aitenant_lacks_praxis_cleanup_finalizer(aitenant=ready_aitenant_without_praxis_annotation)

    @pytest.mark.tier2
    def test_aitenant_non_praxis_annotation_value_uses_legacy_ipp(
        self,
        admin_client: DynamicClient,
        aitenant_infra_namespace: str,
        teardown_resources: bool,
    ) -> None:
        """Given payload-processing-type is not praxis, when bootstrap completes,
        then the value is stored but Praxis cleanup is not enabled.
        """
        with praxis_aitenant_with_bootstrap_gateway(
            admin_client=admin_client,
            cr_namespace=aitenant_infra_namespace,
            payload_processing_type="foo",
            teardown=teardown_resources,
        ) as aitenant:
            deploy_and_verify_aitenant_ready(aitenant=aitenant)
            verify_aitenant_payload_processing_annotation(aitenant=aitenant, expected_value="foo")
            verify_aitenant_lacks_praxis_cleanup_finalizer(aitenant=aitenant)

    @pytest.mark.tier2
    def test_aitenant_empty_annotation_value_uses_legacy_ipp(
        self,
        admin_client: DynamicClient,
        aitenant_infra_namespace: str,
        teardown_resources: bool,
    ) -> None:
        """Given payload-processing-type is empty, when bootstrap completes,
        then the empty value is stored but Praxis cleanup is not enabled.
        """
        with praxis_aitenant_with_bootstrap_gateway(
            admin_client=admin_client,
            cr_namespace=aitenant_infra_namespace,
            payload_processing_type="",
            teardown=teardown_resources,
        ) as aitenant:
            deploy_and_verify_aitenant_ready(aitenant=aitenant)
            verify_aitenant_payload_processing_annotation(aitenant=aitenant, expected_value="")
            verify_aitenant_lacks_praxis_cleanup_finalizer(aitenant=aitenant)

    @pytest.mark.tier2
    def test_aitenant_annotation_value_ipp_uses_legacy_ipp(
        self,
        admin_client: DynamicClient,
        aitenant_infra_namespace: str,
        teardown_resources: bool,
    ) -> None:
        """Given payload-processing-type ipp, when bootstrap completes,
        then the value is stored but Praxis cleanup is not enabled.
        """
        with praxis_aitenant_with_bootstrap_gateway(
            admin_client=admin_client,
            cr_namespace=aitenant_infra_namespace,
            payload_processing_type="ipp",
            teardown=teardown_resources,
        ) as aitenant:
            deploy_and_verify_aitenant_ready(aitenant=aitenant)
            verify_aitenant_payload_processing_annotation(aitenant=aitenant, expected_value="ipp")
            verify_aitenant_lacks_praxis_cleanup_finalizer(aitenant=aitenant)
