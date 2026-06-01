import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace

from tests.model_serving.maas_billing.multitenancy.aigateway.utils import (
    AIGATEWAY_BOOTSTRAPPED_TENANT_NAME,
    AIGATEWAY_NAME_ANNOTATION,
    AIGatewayTestContext,
    aigateway_from_spec,
    build_aigateway_spec,
    deploy_and_verify_aigateway_ready,
    tenant_namespace_name_for_aigateway,
    verify_aigateway_bootstrap_children,
    verify_aigateway_ready,
    verify_bootstrapped_tenant_oidc,
    verify_gateway_https_listener_tls,
    verify_gateway_listener_hostname,
)
from utilities.general import generate_random_name
from utilities.resources.aigateway import AIGateway

LOGGER = structlog.get_logger(name=__name__)

TEST_OIDC_SPEC = {
    "issuerUrl": "https://sso.example.com/realms/maas-aigw-e2e",
    "clientId": "maas-aigw-e2e",
    "ttl": 600,
}


@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest", "aigateway_infra_namespace")
class TestAIGatewayBootstrapFeatures:
    """Check AIGateway settings for namespace adopt, domain, TLS, and OIDC on the new tenant."""

    @pytest.mark.tier1
    def test_aigateway_reconcile_is_idempotent(
        self,
        admin_client: DynamicClient,
        aigateway_for_test: AIGatewayTestContext,
    ) -> None:
        """Verify a bootstrapped AIGateway stays ready after reconcile is re-checked."""
        aigateway = aigateway_for_test["aigateway"]
        verify_aigateway_ready(aigateway=aigateway)
        verify_aigateway_bootstrap_children(
            admin_client=admin_client,
            test_context=aigateway_for_test,
        )
        refreshed_aigateway = AIGateway(
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
        teardown_resources: bool,
        aigateway_infra_namespace: str,
    ) -> None:
        """Verify AIGateway adopts an existing tenant namespace when create is false."""
        aigateway_name = f"e2e-aigw-adopt-ns-{generate_random_name()}"
        tenant_namespace_name = tenant_namespace_name_for_aigateway(aigateway_name=aigateway_name)
        with Namespace(client=admin_client, name=tenant_namespace_name, teardown=teardown_resources) as ns:
            if not ns.exists:
                ns.deploy()
            aigateway_spec = build_aigateway_spec(
                aigateway_name=aigateway_name,
                tenant_namespace_name=tenant_namespace_name,
                create_tenant_namespace=False,
            )
            with aigateway_from_spec(
                admin_client=admin_client,
                aigateway_name=aigateway_name,
                cr_namespace=aigateway_infra_namespace,
                aigateway_spec=aigateway_spec,
                teardown=teardown_resources,
            ) as aigateway:
                deploy_and_verify_aigateway_ready(aigateway=aigateway)
                tenant_namespace = Namespace(
                    client=admin_client,
                    name=tenant_namespace_name,
                    ensure_exists=True,
                )
                annotations = tenant_namespace.instance.metadata.annotations or {}
                assert annotations.get(AIGATEWAY_NAME_ANNOTATION) == aigateway_name

    @pytest.mark.tier2
    def test_aigateway_domain_creates_http_listener_with_hostname(
        self,
        admin_client: DynamicClient,
        teardown_resources: bool,
        aigateway_infra_namespace: str,
    ) -> None:
        """Verify spec.domain configures an HTTP Gateway listener with the expected hostname."""
        aigateway_name = f"e2e-aigw-domain-{generate_random_name()}"
        tenant_domain = f"{aigateway_name}.maas-aigw.test"
        aigateway_spec = build_aigateway_spec(aigateway_name=aigateway_name, domain=tenant_domain)
        with aigateway_from_spec(
            admin_client=admin_client,
            aigateway_name=aigateway_name,
            cr_namespace=aigateway_infra_namespace,
            aigateway_spec=aigateway_spec,
            teardown=teardown_resources,
        ) as aigateway:
            deploy_and_verify_aigateway_ready(aigateway=aigateway)
            verify_gateway_listener_hostname(
                admin_client=admin_client,
                gateway_name=aigateway_name,
                expected_hostname=tenant_domain,
            )

    @pytest.mark.tier2
    def test_aigateway_domain_with_tls_creates_https_listener(
        self,
        admin_client: DynamicClient,
        teardown_resources: bool,
        aigateway_infra_namespace: str,
    ) -> None:
        """Verify spec.domain and spec.tls configure an HTTPS Gateway listener with TLS cert ref."""
        aigateway_name = f"e2e-aigw-tls-{generate_random_name()}"
        tenant_domain = f"{aigateway_name}.maas-aigw.test"
        certificate_secret_name = f"{aigateway_name}-tls"
        aigateway_spec = build_aigateway_spec(
            aigateway_name=aigateway_name,
            domain=tenant_domain,
            tls={"certificateRef": {"name": certificate_secret_name}},
        )
        with aigateway_from_spec(
            admin_client=admin_client,
            aigateway_name=aigateway_name,
            cr_namespace=aigateway_infra_namespace,
            aigateway_spec=aigateway_spec,
            teardown=teardown_resources,
        ) as aigateway:
            deploy_and_verify_aigateway_ready(aigateway=aigateway)
            verify_gateway_https_listener_tls(
                admin_client=admin_client,
                gateway_name=aigateway_name,
                certificate_secret_name=certificate_secret_name,
            )

    @pytest.mark.tier2
    def test_aigateway_oidc_mirrored_to_bootstrapped_tenant(
        self,
        admin_client: DynamicClient,
        teardown_resources: bool,
        aigateway_infra_namespace: str,
    ) -> None:
        """Verify spec.oidc is mirrored to Tenant/default-tenant externalOIDC."""
        aigateway_name = f"e2e-aigw-oidc-{generate_random_name()}"
        tenant_namespace_name = tenant_namespace_name_for_aigateway(aigateway_name=aigateway_name)
        aigateway_spec = build_aigateway_spec(aigateway_name=aigateway_name, oidc=TEST_OIDC_SPEC)
        with aigateway_from_spec(
            admin_client=admin_client,
            aigateway_name=aigateway_name,
            cr_namespace=aigateway_infra_namespace,
            aigateway_spec=aigateway_spec,
            teardown=teardown_resources,
        ) as aigateway:
            deploy_and_verify_aigateway_ready(aigateway=aigateway)
            verify_bootstrapped_tenant_oidc(
                admin_client=admin_client,
                tenant_namespace_name=tenant_namespace_name,
                expected_oidc=TEST_OIDC_SPEC,
            )
            LOGGER.info(
                f"AIGateway oidc mirrored to Tenant/{AIGATEWAY_BOOTSTRAPPED_TENANT_NAME} in '{tenant_namespace_name}'"
            )
