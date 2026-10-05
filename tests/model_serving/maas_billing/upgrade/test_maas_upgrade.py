import pytest
from ocp_resources.gateway_gateway_networking_k8s_io import Gateway
from ocp_resources.maas_auth_policy import MaaSAuthPolicy
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.maas_subscription import MaaSSubscription

from tests.model_serving.maas_billing.upgrade.utils import (
    verify_maas_auth_policy_exists,
    verify_maas_model_ref_exists,
    verify_maas_subscription_ready,
)
from tests.model_serving.maas_billing.utils import (
    MaaSTenantResource,
    verify_maas_gateway_programmed,
    verify_maas_tenant_ready,
)


@pytest.mark.usefixtures("capture_maas_upgrade_baseline")
@pytest.mark.pre_upgrade
class TestPreUpgradeMaaS:
    """Deploy and verify MaaS control plane state before a 3.4 to 3.5 upgrade.

    Steps:
        1. Verify MaaS Gateway is Programmed.
        2. Verify default-tenant legacy Tenant CR is Ready.
        3. Verify MaaSModelRef was created successfully.
        4. Verify MaaSAuthPolicy was created successfully.
        5. Verify MaaSSubscription exists.
        6. Capture state snapshot to ConfigMap for post-upgrade comparison on 3.5.
    """

    def test_maas_gateway_programmed(
        self,
        maas_upgrade_gateway: Gateway,
    ) -> None:
        """Verify MaaS gateway is Programmed before upgrade."""
        verify_maas_gateway_programmed(gateway=maas_upgrade_gateway)

    def test_maas_tenant_ready(
        self,
        maas_upgrade_tenant: MaaSTenantResource,
    ) -> None:
        """Verify default-tenant legacy Tenant CR is Ready before upgrade."""
        verify_maas_tenant_ready(tenant_resource=maas_upgrade_tenant)

    def test_maas_model_ref_created(
        self,
        maas_upgrade_model_ref: MaaSModelRef,
    ) -> None:
        """Verify MaaSModelRef is created before upgrade."""
        verify_maas_model_ref_exists(model_ref=maas_upgrade_model_ref)

    def test_maas_auth_policy_created(
        self,
        maas_upgrade_auth_policy: MaaSAuthPolicy,
    ) -> None:
        """Verify MaaSAuthPolicy is created before upgrade."""
        verify_maas_auth_policy_exists(auth_policy=maas_upgrade_auth_policy)

    def test_maas_subscription_ready(
        self,
        maas_upgrade_subscription: MaaSSubscription,
    ) -> None:
        """Verify MaaSSubscription exists before upgrade."""
        verify_maas_subscription_ready(subscription=maas_upgrade_subscription)
