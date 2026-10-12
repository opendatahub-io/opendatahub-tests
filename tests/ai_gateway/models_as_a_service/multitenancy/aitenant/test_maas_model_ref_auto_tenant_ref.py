"""MaaSModelRef auto-resolve tenantRef tests (RHOAIENG-98879)."""

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.maas_model_ref import MaaSModelRef

from tests.ai_gateway.models_as_a_service.multitenancy.aitenant.utils import (
    AITenantTestContext,
    build_tenant_namespace_discovery_case,
    read_maas_model_ref_resolved_tenant_ref,
    tenant_namespace_for_discovery,
    verify_maas_model_ref_spec_omits_tenant_ref,
    verify_maas_model_ref_tenant_ref_resolution,
    wait_for_maas_model_ref_tenant_resolution_failed,
)
from tests.ai_gateway.models_as_a_service.multitenancy.utils import (
    gateway_ref_from_aitenant,
    provision_tenant_model,
)
from tests.ai_gateway.models_as_a_service.utils import bootstrap_gateway_context
from utilities.constants import MAAS_GATEWAY_NAMESPACE
from utilities.general import generate_random_name

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "aitenant_infra_namespace",
    "tenant_namespace_discovery_prerequisites",
)
class TestMaasModelRefAutoTenantRef:
    """Verify MaaSModelRef tenantRef auto-resolution from the backend HTTPRoute gateway."""

    @pytest.mark.tier2
    def test_maas_model_ref_auto_resolves_tenant_ref_from_gateway(
        self,
        aitenant_tenant_local_maas_model_ref: tuple[AITenantTestContext, MaaSModelRef],
    ) -> None:
        """Given a tenant-local model on an AITenant Gateway without spec.tenantRef,
        when maas-controller reconciles the MaaSModelRef,
        then status.resolvedTenantRef matches the owning AITenant.
        """
        test_context, tenant_model_ref = aitenant_tenant_local_maas_model_ref
        aitenant_name = test_context["aitenant_name"]
        verify_maas_model_ref_tenant_ref_resolution(
            model_ref=tenant_model_ref,
            expected_aitenant_name=aitenant_name,
        )

    @pytest.mark.tier2
    def test_maas_model_ref_explicit_tenant_ref_is_honored(
        self,
        admin_client: DynamicClient,
        aitenant_for_test: AITenantTestContext,
        teardown_resources: bool,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        models_s3_bucket_name: str,
        models_s3_bucket_region: str,
        models_s3_bucket_endpoint: str,
    ) -> None:
        """Given spec.tenantRef set to the owning AITenant,
        when maas-controller reconciles the MaaSModelRef,
        then status.resolvedTenantRef matches the explicit value.
        """
        aitenant_name = aitenant_for_test["aitenant_name"]
        tenant_namespace_name = aitenant_for_test["tenant_namespace_name"]
        gateway_name, gateway_namespace = gateway_ref_from_aitenant(aitenant=aitenant_for_test["aitenant"])
        explicit_model_name = f"e2e-explicit-tenant-ref-{generate_random_name()[:8]}"
        with provision_tenant_model(
            admin_client=admin_client,
            model_name=explicit_model_name,
            tenant_namespace_name=tenant_namespace_name,
            gateway_name=gateway_name,
            gateway_namespace=gateway_namespace,
            aws_access_key=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_s3_region=models_s3_bucket_region,
            aws_s3_bucket=models_s3_bucket_name,
            aws_s3_endpoint=models_s3_bucket_endpoint,
            teardown=teardown_resources,
            spec_tenant_ref=aitenant_name,
        ) as tenant_model_ref:
            verify_maas_model_ref_tenant_ref_resolution(
                model_ref=tenant_model_ref,
                expected_aitenant_name=aitenant_name,
                expected_spec_tenant_ref=aitenant_name,
            )

    @pytest.mark.tier2
    def test_maas_model_ref_tenant_resolution_fails_without_owning_aitenant(
        self,
        admin_client: DynamicClient,
        teardown_resources: bool,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        models_s3_bucket_name: str,
        models_s3_bucket_region: str,
        models_s3_bucket_endpoint: str,
    ) -> None:
        """Given a tenant-local model on a Gateway with no owning AITenant and HTTPRoute accepted,
        when maas-controller reconciles MaaSModelRef without spec.tenantRef,
        then phase is Failed, resolvedTenantRef is unset, and Ready mentions missing AITenant.
        """
        discovery_case = build_tenant_namespace_discovery_case()
        tenant_namespace_name = discovery_case["tenant_namespace_name"]
        orphan_gateway_name = f"e2e-orphan-gw-{discovery_case['suffix']}"
        model_name = f"e2e-unresolved-tenant-{discovery_case['suffix']}"
        with (
            tenant_namespace_for_discovery(
                admin_client=admin_client,
                tenant_namespace_name=tenant_namespace_name,
                teardown=teardown_resources,
            ),
            bootstrap_gateway_context(
                admin_client=admin_client,
                gateway_name=orphan_gateway_name,
                gateway_namespace=MAAS_GATEWAY_NAMESPACE,
                teardown=teardown_resources,
            ),
            provision_tenant_model(
                admin_client=admin_client,
                model_name=model_name,
                tenant_namespace_name=tenant_namespace_name,
                gateway_name=orphan_gateway_name,
                gateway_namespace=MAAS_GATEWAY_NAMESPACE,
                aws_access_key=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_s3_region=models_s3_bucket_region,
                aws_s3_bucket=models_s3_bucket_name,
                aws_s3_endpoint=models_s3_bucket_endpoint,
                teardown=teardown_resources,
            ) as unresolved_model_ref,
        ):
            verify_maas_model_ref_spec_omits_tenant_ref(model_ref=unresolved_model_ref)
            failure_message = wait_for_maas_model_ref_tenant_resolution_failed(
                model_ref=unresolved_model_ref,
            )
            resolved_tenant_ref = read_maas_model_ref_resolved_tenant_ref(model_ref=unresolved_model_ref)
            assert resolved_tenant_ref is None, (
                f"Expected empty resolvedTenantRef after tenant resolution failure, got {resolved_tenant_ref!r}"
            )
            assert "aitenant" in failure_message.lower(), (
                f"Expected Ready message to mention missing AITenant, got {failure_message!r}"
            )
            LOGGER.info(
                f"MaaSModelRef '{tenant_namespace_name}/{model_name}' "
                f"failed tenant resolution as expected: {failure_message}"
            )
