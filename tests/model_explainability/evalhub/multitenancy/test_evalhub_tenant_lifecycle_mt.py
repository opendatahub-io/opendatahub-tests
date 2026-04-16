import uuid

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.resource import ResourceEditor
from ocp_resources.role_binding import RoleBinding
from ocp_resources.service_account import ServiceAccount
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_explainability.evalhub.constants import (
    EVALHUB_MT_CR_NAME,
    EVALHUB_TENANT_LABEL_KEY,
)
from tests.model_explainability.evalhub.utils import tenant_rbac_absent, tenant_rbac_ready
from utilities.infra import create_ns

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-tenant-lifecycle"},
        ),
    ],
    indirect=True,
)
@pytest.mark.model_explainability
class TestEvalHubTenantLifecycle:
    """Tests for operator tenant RBAC lifecycle (provision and cleanup)."""

    def test_operator_cleans_up_rbac_on_label_removal(
        self,
        admin_client: DynamicClient,
        evalhub_mt_deployment: Deployment,
    ) -> None:
        """When the tenant label is removed from a namespace, the operator should
        delete the RoleBindings, ServiceAccount, and service-CA ConfigMap it provisioned.
        """
        suffix = uuid.uuid4().hex[:6]
        ns_name = f"test-evalhub-lifecycle-{suffix}"

        with create_ns(
            admin_client=admin_client,
            name=ns_name,
            labels={EVALHUB_TENANT_LABEL_KEY: "true"},
        ) as ns:
            # Wait for operator to provision RBAC
            try:
                for ready in TimeoutSampler(
                    wait_timeout=120,
                    sleep=5,
                    func=tenant_rbac_ready,
                    admin_client=admin_client,
                    namespace=ns.name,
                ):
                    if ready:
                        LOGGER.info(f"Operator RBAC provisioned in {ns.name}")
                        break
            except TimeoutExpiredError:
                pytest.fail(f"Operator did not provision RBAC in namespace '{ns.name}' within timeout")

            # Remove the tenant label
            ResourceEditor(patches={ns: {"metadata": {"labels": {EVALHUB_TENANT_LABEL_KEY: None}}}}).update()
            LOGGER.info(f"Removed tenant label from {ns.name}")

            # Wait for operator to clean up RBAC
            try:
                for absent in TimeoutSampler(
                    wait_timeout=120,
                    sleep=5,
                    func=tenant_rbac_absent,
                    admin_client=admin_client,
                    namespace=ns.name,
                ):
                    if absent:
                        LOGGER.info(f"Operator RBAC cleaned up in {ns.name}")
                        break
            except TimeoutExpiredError:
                pytest.fail(f"Operator did not clean up RBAC in namespace '{ns.name}' within timeout")

            # Verify nothing operator-managed remains
            rbs = [
                rb
                for rb in RoleBinding.get(client=admin_client, namespace=ns.name)
                if rb.name.startswith(EVALHUB_MT_CR_NAME)
            ]
            assert rbs == [], f"Expected no operator RoleBindings, found: {[rb.name for rb in rbs]}"

            sas = [
                sa
                for sa in ServiceAccount.get(client=admin_client, namespace=ns.name)
                if sa.name.startswith(EVALHUB_MT_CR_NAME) and "job" in sa.name
            ]
            assert sas == [], f"Expected no operator ServiceAccounts, found: {[sa.name for sa in sas]}"

            cms = [
                cm
                for cm in ConfigMap.get(client=admin_client, namespace=ns.name)
                if cm.name.startswith(EVALHUB_MT_CR_NAME) and "service-ca" in cm.name
            ]
            assert cms == [], f"Expected no operator ConfigMaps, found: {[cm.name for cm in cms]}"
