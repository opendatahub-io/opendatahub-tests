import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import DynamicApiError

from tests.model_serving.maas_billing.maas_subscription.utils import MAAS_SUBSCRIPTION_NAMESPACE
from tests.model_serving.maas_billing.multitenancy.aitenant.utils import (
    AITENANT_INFRA_NAMESPACE,
    AITenantTestContext,
    aitenant_from_spec,
    build_aitenant_spec,
    verify_derived_tenant_namespace_name,
)

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest", "aitenant_infra_namespace")
class TestAITenantInvalidNamespacePlacement:
    """Tier3 tests for AITenant invalid namespace placement and derived tenant namespace naming."""

    @pytest.mark.tier3
    @pytest.mark.parametrize(
        "aitenant_outside_infra_namespace",
        [
            pytest.param("applications_namespace", id="test_cr_in_applications_namespace"),
            pytest.param("legacy_tenant_namespace", id="test_cr_in_legacy_tenant_namespace"),
        ],
        indirect=True,
    )
    def test_aitenant_rejected_outside_infra_namespace(
        self,
        admin_client: DynamicClient,
        teardown_resources: bool,
        aitenant_outside_infra_namespace: tuple[str, str],
    ) -> None:
        """Verify the admission webhook rejects AITenant CRs outside ai-tenants."""
        aitenant_name, cr_namespace = aitenant_outside_infra_namespace
        aitenant_spec = build_aitenant_spec(aitenant_name=aitenant_name)
        aitenant = aitenant_from_spec(
            admin_client=admin_client,
            aitenant_name=aitenant_name,
            cr_namespace=cr_namespace,
            aitenant_spec=aitenant_spec,
            teardown=teardown_resources,
        )
        with pytest.raises(DynamicApiError, match=r"(?i)admission webhook"), aitenant:
            aitenant.deploy()
        assert not aitenant.exists, f"AITenant '{aitenant_name}' should not exist in '{cr_namespace}'"
        LOGGER.info(f"AITenant '{aitenant_name}' rejected outside '{AITENANT_INFRA_NAMESPACE}' as expected")

    @pytest.mark.tier3
    def test_aitenant_derives_non_default_tenant_namespace(
        self,
        aitenant_derived_namespace_case: tuple[AITenantTestContext, str],
    ) -> None:
        """Verify non-default AITenant status.tenantNamespace is ai-tenant-{name}, not models-as-a-service."""
        test_context, expected_tenant_namespace = aitenant_derived_namespace_case
        assert expected_tenant_namespace != MAAS_SUBSCRIPTION_NAMESPACE
        verify_derived_tenant_namespace_name(
            aitenant=test_context["aitenant"],
            expected_tenant_namespace_name=expected_tenant_namespace,
        )
        LOGGER.info(
            f"AITenant '{test_context['aitenant_name']}' derived tenant namespace "
            f"'{expected_tenant_namespace}' as expected"
        )
