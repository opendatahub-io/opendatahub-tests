from typing import Any, TypedDict

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.gateway_gateway_networking_k8s_io import Gateway
from ocp_resources.namespace import Namespace

from tests.model_serving.maas_billing.maas_subscription.utils import MAAS_SUBSCRIPTION_NAMESPACE
from tests.model_serving.maas_billing.utils import verify_maas_gateway_programmed, verify_maas_tenant_ready
from utilities.constants import MAAS_GATEWAY_NAMESPACE, ApiGroups
from utilities.resources.aigateway import AIGateway
from utilities.resources.tenant import Tenant

LOGGER = structlog.get_logger(name=__name__)

AIGATEWAY_CRD_NAME = f"aigateways.{ApiGroups.MAAS_IO}"
AIGATEWAY_INFRA_NAMESPACE = "ai-gateway-system"
AIGATEWAY_BOOTSTRAPPED_TENANT_NAME = "default-tenant"
AIGATEWAY_TENANT_NAMESPACE_SUFFIX = "-maas"


class AIGatewayTestContext(TypedDict):
    aigateway: AIGateway
    aigateway_name: str
    tenant_namespace_name: str


def tenant_namespace_name_for_aigateway(aigateway_name: str) -> str:
    """Derive the tenant namespace name created for an AIGateway."""
    return f"{aigateway_name}{AIGATEWAY_TENANT_NAMESPACE_SUFFIX}"


def build_aigateway_spec(
    aigateway_name: str,
    cleanup_on_delete: bool = True,
    create_tenant_namespace: bool = True,
) -> dict[str, Any]:
    """Build a minimal AIGateway spec for bootstrap testing."""
    return {
        "tenantNamespace": {
            "name": tenant_namespace_name_for_aigateway(aigateway_name=aigateway_name),
            "create": create_tenant_namespace,
            "cleanupOnDelete": cleanup_on_delete,
        },
        "gateway": {
            "namespace": MAAS_GATEWAY_NAMESPACE,
            "gatewayClassName": "openshift-default",
        },
    }


def build_aigateway_test_context(aigateway: AIGateway) -> AIGatewayTestContext:
    """Build the standard test context dict from a deployed AIGateway."""
    return AIGatewayTestContext(
        aigateway=aigateway,
        aigateway_name=aigateway.name,
        tenant_namespace_name=tenant_namespace_name_for_aigateway(aigateway_name=aigateway.name),
    )


def verify_aigateway_ready(aigateway: AIGateway) -> None:
    """Assert the AIGateway exists and reports Ready=True with phase Active."""
    assert aigateway.exists, f"AIGateway '{aigateway.name}' not found in namespace '{aigateway.namespace}'"
    aigateway.wait_for_condition(condition="Ready", status="True", timeout=300)
    phase = getattr(aigateway.instance.status, "phase", "") or ""
    assert phase == "Active", f"Expected AIGateway phase Active, got '{phase}'"


def verify_aigateway_bootstrap_children(
    admin_client: DynamicClient,
    test_context: AIGatewayTestContext,
) -> None:
    """Assert AIGateway reconciliation created the expected child resources."""
    aigateway_name = test_context["aigateway_name"]
    tenant_namespace_name = test_context["tenant_namespace_name"]

    tenant_namespace = Namespace(
        client=admin_client,
        name=tenant_namespace_name,
        ensure_exists=True,
    )
    assert tenant_namespace.exists, f"Tenant namespace '{tenant_namespace_name}' was not created"

    tenant_gateway = Gateway(
        client=admin_client,
        name=aigateway_name,
        namespace=MAAS_GATEWAY_NAMESPACE,
        ensure_exists=True,
    )
    verify_maas_gateway_programmed(gateway=tenant_gateway)

    bootstrapped_tenant = Tenant(
        client=admin_client,
        name=AIGATEWAY_BOOTSTRAPPED_TENANT_NAME,
        namespace=tenant_namespace_name,
    )
    verify_maas_tenant_ready(tenant=bootstrapped_tenant)
    LOGGER.info(
        f"AIGateway '{aigateway_name}' bootstrap verified: namespace, gateway, and "
        f"Tenant/{AIGATEWAY_BOOTSTRAPPED_TENANT_NAME} are ready"
    )


def verify_default_maas_tenant_unaffected(admin_client: DynamicClient) -> None:
    """Assert the cluster default-tenant in models-as-a-service is still Ready."""
    default_tenant = Tenant(
        client=admin_client,
        name=AIGATEWAY_BOOTSTRAPPED_TENANT_NAME,
        namespace=MAAS_SUBSCRIPTION_NAMESPACE,
    )
    verify_maas_tenant_ready(tenant=default_tenant)
    LOGGER.info(
        f"Regression check passed: Tenant/{AIGATEWAY_BOOTSTRAPPED_TENANT_NAME} in "
        f"'{MAAS_SUBSCRIPTION_NAMESPACE}' is still Ready"
    )
