import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import NotFoundError, ResourceNotFoundError
from ocp_resources.deployment import Deployment
from timeout_sampler import TimeoutSampler

from tests.model_serving.maas_billing.multitenancy.aitenant.utils import (
    AIGATEWAY_BOOTSTRAPPED_TENANT_NAME,
)
from utilities.resources.aitenant import AITenant
from utilities.resources.http_route import HTTPRoute
from utilities.resources.tenant import Tenant

LOGGER = structlog.get_logger(name=__name__)

MAAS_API_DEPLOYMENT_NAME = "maas-api"


def maas_api_deployment_name_for_aitenant(aitenant_name: str) -> str:
    """Return the per-tenant maas-api Deployment name for an additional AITenant."""
    return f"{MAAS_API_DEPLOYMENT_NAME}-{aitenant_name}"


def maas_api_route_name_for_aitenant(aitenant_name: str) -> str:
    """Return the per-tenant maas-api HTTPRoute name applied by the Tenant reconciler post-render."""
    return f"{MAAS_API_DEPLOYMENT_NAME}-{aitenant_name}-route"


def gateway_ref_from_aitenant(aitenant: AITenant) -> tuple[str, str]:
    """Return the Gateway name and namespace referenced by AITenant status."""
    fresh_aitenant = AITenant(
        client=aitenant.client,
        name=aitenant.name,
        namespace=aitenant.namespace,
        wait_for_resource=False,
    )
    status_gateway_ref = getattr(fresh_aitenant.instance.status, "gatewayRef", None)
    assert status_gateway_ref is not None, f"AITenant '{aitenant.name}' status.gatewayRef should be set after bootstrap"
    return status_gateway_ref.name, status_gateway_ref.namespace


def wait_for_bootstrapped_tenant_deployments_available(
    admin_client: DynamicClient,
    tenant_namespace_name: str,
    timeout: int = 300,
) -> Tenant:
    """Wait until the bootstrapped Tenant reports DeploymentsAvailable=True."""
    bootstrapped_tenant = Tenant(
        client=admin_client,
        name=AIGATEWAY_BOOTSTRAPPED_TENANT_NAME,
        namespace=tenant_namespace_name,
        ensure_exists=True,
    )
    bootstrapped_tenant.wait_for_condition(
        condition="DeploymentsAvailable",
        status="True",
        timeout=timeout,
    )
    return bootstrapped_tenant


def verify_maas_api_deployment_for_aitenant(
    admin_client: DynamicClient,
    applications_namespace: str,
    aitenant_name: str,
    tenant_namespace_name: str,
) -> None:
    """Assert the per-tenant maas-api Deployment is Available in the applications namespace."""
    wait_for_bootstrapped_tenant_deployments_available(
        admin_client=admin_client,
        tenant_namespace_name=tenant_namespace_name,
    )
    deployment_name = maas_api_deployment_name_for_aitenant(aitenant_name=aitenant_name)
    maas_api_deployment = Deployment(
        client=admin_client,
        name=deployment_name,
        namespace=applications_namespace,
        ensure_exists=True,
    )
    assert maas_api_deployment.exists, f"Deployment/{deployment_name} not found in namespace '{applications_namespace}'"
    maas_api_deployment.wait_for_condition(condition="Available", status="True", timeout=300)
    LOGGER.info(f"Deployment/{deployment_name} is Available in applications namespace '{applications_namespace}'")


def get_maas_api_httproute(
    admin_client: DynamicClient,
    route_name: str,
    route_namespace: str,
) -> HTTPRoute:
    """Look up a per-tenant maas-api HTTPRoute; raise when it is not present yet."""
    return HTTPRoute(
        client=admin_client,
        name=route_name,
        namespace=route_namespace,
        wait_for_resource=False,
        ensure_exists=True,
    )


def wait_for_maas_api_httproute(
    admin_client: DynamicClient,
    route_name: str,
    route_namespace: str,
    timeout: int = 300,
) -> HTTPRoute:
    """Poll until the per-tenant maas-api HTTPRoute exists."""
    for route in TimeoutSampler(
        wait_timeout=timeout,
        sleep=3,
        func=get_maas_api_httproute,
        exceptions_dict={NotFoundError: [], ResourceNotFoundError: []},
        admin_client=admin_client,
        route_name=route_name,
        route_namespace=route_namespace,
    ):
        return route


def httproute_references_gateway(
    route: HTTPRoute,
    gateway_name: str,
    gateway_namespace: str,
) -> bool:
    """Return True when the HTTPRoute parentRefs include the expected Gateway."""
    route_namespace = route.namespace
    parent_refs = getattr(route.instance.spec, "parentRefs", None) or []
    for parent_ref in parent_refs:
        parent_kind = getattr(parent_ref, "kind", None) or "Gateway"
        if parent_kind != "Gateway":
            continue
        parent_name = getattr(parent_ref, "name", None)
        parent_namespace = getattr(parent_ref, "namespace", None) or route_namespace
        if parent_name == gateway_name and parent_namespace == gateway_namespace:
            return True
    return False


def verify_maas_api_httproute_attached_to_gateway(
    admin_client: DynamicClient,
    applications_namespace: str,
    aitenant_name: str,
    tenant_namespace_name: str,
    gateway_name: str,
    gateway_namespace: str,
    timeout: int = 300,
) -> None:
    """Assert the per-tenant maas-api HTTPRoute exists in the applications namespace.

    Also assert the route attaches to the tenant Gateway via parentRefs.
    """
    wait_for_bootstrapped_tenant_deployments_available(
        admin_client=admin_client,
        tenant_namespace_name=tenant_namespace_name,
        timeout=timeout,
    )
    route_name = maas_api_route_name_for_aitenant(
        aitenant_name=aitenant_name,
    )
    maas_api_route = wait_for_maas_api_httproute(
        admin_client=admin_client,
        route_name=route_name,
        route_namespace=applications_namespace,
        timeout=timeout,
    )
    assert httproute_references_gateway(
        route=maas_api_route,
        gateway_name=gateway_name,
        gateway_namespace=gateway_namespace,
    ), (
        f"HTTPRoute/{route_name} in '{applications_namespace}' should reference "
        f"Gateway '{gateway_namespace}/{gateway_name}' in parentRefs"
    )
    LOGGER.info(
        f"HTTPRoute/{route_name} in '{applications_namespace}' is attached to "
        f"Gateway '{gateway_namespace}/{gateway_name}'"
    )
