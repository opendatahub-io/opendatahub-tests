import pytest
from kubernetes.dynamic import DynamicClient
from pytest_testconfig import config as py_config

from tests.model_serving.maas_billing.multitenancy.aitenant.utils import AITenantTestContext
from tests.model_serving.maas_billing.multitenancy.utils import (
    gateway_ref_from_aitenant,
    verify_maas_api_deployment_for_aitenant,
    verify_maas_api_httproute_attached_to_gateway,
)


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "aitenant_infra_namespace",
)
class TestMultiTenantMaaSApi:
    """Verify per-tenant maas-api infrastructure after AITenant bootstrap."""

    @pytest.mark.tier2
    def test_aitenant_deploys_maas_api_in_applications_namespace(
        self,
        admin_client: DynamicClient,
        aitenant_for_test: AITenantTestContext,
    ) -> None:
        """Given a Ready AITenant, when the Tenant reconciler finishes,
        then maas-api-{tenant} is Available in the applications namespace.
        """
        verify_maas_api_deployment_for_aitenant(
            admin_client=admin_client,
            applications_namespace=py_config["applications_namespace"],
            aitenant_name=aitenant_for_test["aitenant_name"],
            tenant_namespace_name=aitenant_for_test["tenant_namespace_name"],
        )

    @pytest.mark.tier2
    def test_aitenant_maas_api_httproute_attached_to_gateway(
        self,
        admin_client: DynamicClient,
        aitenant_for_test: AITenantTestContext,
    ) -> None:
        """Given a Ready AITenant with a programmed Gateway, when checking maas-api-{tenant}-route,
        then parentRefs target that Gateway.
        """
        gateway_name, gateway_namespace = gateway_ref_from_aitenant(aitenant=aitenant_for_test["aitenant"])
        verify_maas_api_httproute_attached_to_gateway(
            admin_client=admin_client,
            applications_namespace=py_config["applications_namespace"],
            aitenant_name=aitenant_for_test["aitenant_name"],
            tenant_namespace_name=aitenant_for_test["tenant_namespace_name"],
            gateway_name=gateway_name,
            gateway_namespace=gateway_namespace,
        )
