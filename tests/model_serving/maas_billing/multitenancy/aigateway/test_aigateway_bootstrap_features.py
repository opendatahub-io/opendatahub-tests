import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace

from tests.model_serving.maas_billing.multitenancy.aigateway.utils import (
    AIGATEWAY_BOOTSTRAPPED_TENANT_NAME,
    AIGATEWAY_NAME_ANNOTATION,
    AIGATEWAY_TEST_OIDC_SPEC,
    AIGatewayTestContext,
    verify_aigateway_bootstrap_children,
    verify_aigateway_ready,
    verify_bootstrapped_tenant_oidc,
)
from utilities.resources.aitenant import AITenant

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest", "aigateway_infra_namespace")
class TestAIGatewayBootstrapFeatures:
    """Check AITenant settings for namespace adopt and OIDC on the new tenant."""

    @pytest.mark.tier1
    def test_aigateway_bootstrap_children_stay_ready(
        self,
        admin_client: DynamicClient,
        aigateway_for_test: AIGatewayTestContext,
    ) -> None:
        """Verify a bootstrapped AITenant reports Ready and creates expected child resources."""
        verify_aigateway_ready(aigateway=aigateway_for_test["aigateway"])
        verify_aigateway_bootstrap_children(
            admin_client=admin_client,
            test_context=aigateway_for_test,
        )

    @pytest.mark.tier1
    def test_aigateway_stays_ready_after_refetch(
        self,
        admin_client: DynamicClient,
        aigateway_for_test: AIGatewayTestContext,
    ) -> None:
        """Verify a bootstrapped AITenant stays ready after reconcile is re-checked from a fresh client."""
        aigateway = aigateway_for_test["aigateway"]
        refreshed_aigateway = AITenant(
            client=admin_client,
            name=aigateway.name,
            namespace=aigateway.namespace,
            wait_for_resource=False,
        )
        verify_aigateway_ready(aigateway=refreshed_aigateway)
        verify_aigateway_bootstrap_children(
            admin_client=admin_client,
            test_context=aigateway_for_test,
        )

    @pytest.mark.tier2
    def test_aigateway_adopts_preexisting_namespace_when_create_disabled(
        self,
        admin_client: DynamicClient,
        aigateway_adopting_preexisting_namespace: AIGatewayTestContext,
    ) -> None:
        """Verify AITenant adopts an existing tenant namespace when create is false."""
        test_context = aigateway_adopting_preexisting_namespace
        tenant_namespace = Namespace(
            client=admin_client,
            name=test_context["tenant_namespace_name"],
            ensure_exists=True,
        )
        annotations = dict(tenant_namespace.instance.metadata.annotations or {})
        assert annotations.get(AIGATEWAY_NAME_ANNOTATION) == test_context["aigateway_name"]

    @pytest.mark.tier2
    def test_aigateway_oidc_mirrored_to_bootstrapped_tenant(
        self,
        admin_client: DynamicClient,
        aigateway_with_oidc: AIGatewayTestContext,
    ) -> None:
        """Verify spec.oidc is mirrored to Tenant/default-tenant externalOIDC."""
        verify_bootstrapped_tenant_oidc(
            admin_client=admin_client,
            tenant_namespace_name=aigateway_with_oidc["tenant_namespace_name"],
            expected_oidc=AIGATEWAY_TEST_OIDC_SPEC,
        )
        LOGGER.info(
            f"AITenant oidc mirrored to Tenant/{AIGATEWAY_BOOTSTRAPPED_TENANT_NAME} in "
            f"'{aigateway_with_oidc['tenant_namespace_name']}'"
        )
