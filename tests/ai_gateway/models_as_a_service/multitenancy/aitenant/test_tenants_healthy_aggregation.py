"""MaaS Config TenantsHealthy aggregation tests (RHOAIENG-98881)."""

import pytest
import structlog

from tests.ai_gateway.models_as_a_service.multitenancy.aitenant.utils import (
    AITENANT_TENANT_NAMESPACE_FAILED_REASON,
    wait_until_aitenant_status,
)
from tests.ai_gateway.models_as_a_service.multitenancy.utils import (
    wait_for_maas_config_all_tenants_healthy,
    wait_for_maas_config_tenants_unhealthy_listing_tenant,
)
from utilities.resources.aitenant import AITenant
from utilities.resources.maas_config import Config as MaaSConfig

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "aitenant_infra_namespace",
    "maas_config_tenants_healthy_reconcile_enabled",
    "tenants_healthy_aggregation_orphan_cleanup",
)
class TestMaasConfigTenantsHealthyAggregation:
    """Verify maas-controller aggregates AITenant health onto Config/default."""

    @pytest.mark.tier2
    @pytest.mark.usefixtures("ready_aitenant")
    def test_config_tenants_healthy_when_all_aitenants_ready(
        self,
        maas_config_default: MaaSConfig,
    ) -> None:
        """Given every AITenant in the infra namespace is Ready,
        when tenant health aggregation runs,
        then Config/default reports TenantsHealthy=True with reason AllTenantsHealthy.
        """
        wait_for_maas_config_all_tenants_healthy(maas_config=maas_config_default)

    @pytest.mark.tier2
    @pytest.mark.usefixtures("ready_aitenant")
    def test_config_tenants_unhealthy_when_one_aitenant_failed(
        self,
        aitenant_on_namespace_owned_by_other: AITenant,
        maas_config_default: MaaSConfig,
    ) -> None:
        """Given Ready and Failed AITenants in the infra namespace,
        when tenant health aggregation runs,
        then Config/default reports TenantsHealthy=False with reason TenantsDegraded or
        TenantsBlocked and lists the failed tenant in the condition message.
        """
        failed_aitenant = aitenant_on_namespace_owned_by_other
        wait_until_aitenant_status(
            aitenant=failed_aitenant,
            phase="Failed",
            ready_reason=AITENANT_TENANT_NAMESPACE_FAILED_REASON,
        )
        wait_for_maas_config_tenants_unhealthy_listing_tenant(
            maas_config=maas_config_default,
            unhealthy_tenant_name=failed_aitenant.name,
        )
        LOGGER.info(
            f"Config/default TenantsHealthy=False after failed AITenant '{failed_aitenant.name}' "
            "as expected for tenant health aggregation"
        )
