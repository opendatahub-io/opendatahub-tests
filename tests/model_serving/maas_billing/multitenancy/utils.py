from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from typing import Any, TypedDict
from urllib.parse import urlparse

import requests
import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import NotFoundError, ResourceNotFoundError
from ocp_resources.deployment import Deployment
from ocp_resources.gateway_gateway_networking_k8s_io import Gateway
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.maas_auth_policy import MaaSAuthPolicy
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.maas_subscription import MaaSSubscription
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.service_account import ServiceAccount
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_serving.maas_billing.maas_api_key.utils import (
    get_auth_policy_callback_url,
    get_auth_policy_condition,
)
from tests.model_serving.maas_billing.maas_subscription.utils import create_maas_subscription
from tests.model_serving.maas_billing.multitenancy.aitenant.utils import (
    AIGATEWAY_BOOTSTRAPPED_TENANT_NAME,
    AIGATEWAY_GATEWAY_CLASS_NAME,
)
from tests.model_serving.maas_billing.utils import (
    assert_api_key_created_ok,
    create_api_key,
    gateway_probe_reaches_maas_api,
    revoke_api_key,
    verify_maas_gateway_programmed,
)
from utilities.constants import MAAS_GATEWAY_NAMESPACE, ApiGroups, ContainerImages, ModelStorage
from utilities.general import generate_random_name
from utilities.infra import s3_endpoint_secret
from utilities.llmd_utils import create_llmisvc
from utilities.resources.aitenant import AITenant
from utilities.resources.auth_policy import AuthPolicy
from utilities.resources.http_route import HTTPRoute
from utilities.resources.route import Route
from utilities.resources.tenant import Tenant

LOGGER = structlog.get_logger(name=__name__)

MAAS_API_DEPLOYMENT_NAME = "maas-api"
MAAS_HOST_PREFIX = "maas."
TENANT_GATEWAY_SERVICE_SUFFIX = "-openshift-default"
TENANT_GATEWAY_ROUTE_SUFFIX = "-route"
GATEWAY_ACCESS_LABEL_PREFIX = "maas.opendatahub.io/gateway-access-"
GATEWAY_HTTP_PORT = 80
AUTHORINO_TLS_BOOTSTRAP_ANNOTATION = "security.opendatahub.io/authorino-tls-bootstrap"
TENANT_GATEWAY_MAAS_AUTH_POLICY_SUFFIX = "-maas-auth"
SHARED_MAAS_API_SERVICE_NAME = MAAS_API_DEPLOYMENT_NAME
TENANT_ISOLATION_MODEL_NAME = "llm-s3-tinyllama-free"
TENANT_MODELS_BUCKET_SECRET_NAME = "models-bucket-secret"  # pragma: allowlist secret
TENANT_MODELS_BUCKET_SA_NAME = "models-bucket-sa"


class TenantIsolationGovernance(TypedDict):
    aitenant_name: str
    tenant_namespace_name: str
    model_name: str
    model_namespace: str
    auth_policy_name: str
    subscription_name: str


def tenant_isolation_auth_policy_name(aitenant_name: str) -> str:
    """Return the MaaSAuthPolicy name for tenant isolation governance."""
    return f"{aitenant_name}-policy"


def tenant_isolation_subscription_name(aitenant_name: str) -> str:
    """Return the MaaSSubscription name for tenant isolation governance."""
    return f"{aitenant_name}-sub"


def tenant_model_kserve_route_name(model_name: str) -> str:
    """Return the KServe HTTPRoute name for a tenant-local LLMInferenceService."""
    return f"{model_name}-kserve-route"


def ingress_domain_from_maas_host(maas_host: str) -> str:
    """Return the cluster ingress domain from a maas gateway hostname."""
    assert maas_host.startswith(MAAS_HOST_PREFIX), (
        f"Expected maas host to start with {MAAS_HOST_PREFIX!r}, got {maas_host!r}"
    )
    return maas_host.removeprefix(MAAS_HOST_PREFIX)


def tenant_gateway_service_name(gateway_name: str) -> str:
    """Return the OpenShift Service fronting a tenant Gateway."""
    return f"{gateway_name}{TENANT_GATEWAY_SERVICE_SUFFIX}"


def tenant_gateway_route_name(gateway_name: str) -> str:
    """Return the external OpenShift Route name for a tenant Gateway."""
    return f"{gateway_name}{TENANT_GATEWAY_ROUTE_SUFFIX}"


def maas_api_base_url_for_gateway(gateway_name: str, maas_host: str, scheme: str) -> str:
    """Return the external maas-api base URL for a tenant Gateway."""
    ingress_domain = ingress_domain_from_maas_host(maas_host=maas_host)
    return f"{scheme}://{gateway_name}.{ingress_domain}/maas-api"


def wait_for_tenant_gateway_maas_api_reachable(
    request_session_http: requests.Session,
    gateway_name: str,
    maas_host: str,
    wait_timeout: int = 300,
    sleep: int = 5,
    request_timeout_seconds: int = 30,
) -> None:
    """Wait until maas-api responds through a tenant Gateway's external OpenShift Route."""
    base_url = maas_api_base_url_for_gateway(gateway_name=gateway_name, maas_host=maas_host, scheme="https")
    probe_url = f"{base_url}/v1/models"
    for gateway_reachable, status_code, response_text in TimeoutSampler(
        wait_timeout=wait_timeout,
        sleep=sleep,
        func=gateway_probe_reaches_maas_api,
        http_session=request_session_http,
        probe_url=probe_url,
        request_timeout_seconds=request_timeout_seconds,
        exceptions_dict={requests.RequestException: []},
    ):
        if gateway_reachable:
            LOGGER.info(f"Tenant Gateway '{gateway_name}' maas-api reachable at {probe_url} (status={status_code})")
            return
        LOGGER.warning(
            f"Tenant Gateway '{gateway_name}' maas-api not reachable at {probe_url}: "
            f"status={status_code} body={response_text[:200]}"
        )
    raise TimeoutExpiredError(
        f"Timed out waiting for maas-api to respond via tenant Gateway '{gateway_name}' at {probe_url}"
    )


def gateway_access_label_key(gateway_name: str) -> str:
    """Return the namespace label key that grants HTTPRoute attachment to a tenant Gateway."""
    return f"{GATEWAY_ACCESS_LABEL_PREFIX}{gateway_name}"


def gateway_access_namespace_labels(gateway_name: str) -> dict[str, str]:
    """Return namespace labels required for a tenant Gateway to accept HTTPRoutes."""
    return {gateway_access_label_key(gateway_name=gateway_name): "true"}


def tenant_gateway_hostname(gateway_name: str, maas_host: str) -> str:
    """Return the external hostname routed to a tenant Gateway."""
    ingress_domain = ingress_domain_from_maas_host(maas_host=maas_host)
    return f"{gateway_name}.{ingress_domain}"


def isolation_bootstrap_gateway_body(
    gateway_name: str,
    gateway_namespace: str,
) -> dict[str, Any]:
    """Build a tenant Gateway that accepts labeled maas-api HTTPRoutes on HTTP.

    External hostnames are configured on the OpenShift Route, not on the Gateway listener.
    """
    gateway_access_label = gateway_access_label_key(gateway_name=gateway_name)
    return {
        "apiVersion": "gateway.networking.k8s.io/v1",
        "kind": "Gateway",
        "metadata": {
            "name": gateway_name,
            "namespace": gateway_namespace,
            "annotations": {
                AUTHORINO_TLS_BOOTSTRAP_ANNOTATION: "true",
            },
        },
        "spec": {
            "gatewayClassName": AIGATEWAY_GATEWAY_CLASS_NAME,
            "listeners": [
                {
                    "name": "http",
                    "port": GATEWAY_HTTP_PORT,
                    "protocol": "HTTP",
                    "allowedRoutes": {
                        "namespaces": {
                            "from": "Selector",
                            "selector": {
                                "matchLabels": {
                                    gateway_access_label: "true",
                                },
                            },
                        },
                    },
                },
            ],
        },
    }


def label_namespace_gateway_access(
    admin_client: DynamicClient,
    namespace_name: str,
    gateway_name: str,
) -> None:
    """Label a namespace so HTTPRoutes can attach to the tenant Gateway listener."""
    namespace = Namespace(client=admin_client, name=namespace_name, ensure_exists=True)
    assert namespace.exists, f"Namespace '{namespace_name}' not found"
    ResourceEditor(
        patches={namespace: {"metadata": {"labels": gateway_access_namespace_labels(gateway_name=gateway_name)}}},
    ).update()
    LOGGER.info(f"Labeled namespace '{namespace_name}' with {gateway_access_label_key(gateway_name=gateway_name)!r}")


@contextmanager
def isolation_bootstrap_gateway_context(
    admin_client: DynamicClient,
    gateway_name: str,
    applications_namespace: str,
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
    teardown: bool = True,
) -> Generator[Gateway, Any, Any]:
    """Yield a bootstrap Gateway with gateway-access labels and a tenant-scoped HTTP listener."""
    applications_ns = Namespace(client=admin_client, name=applications_namespace, ensure_exists=True)
    assert applications_ns.exists, f"Applications namespace '{applications_namespace}' not found"
    with (
        ResourceEditor(
            patches={
                applications_ns: {
                    "metadata": {"labels": gateway_access_namespace_labels(gateway_name=gateway_name)},
                },
            },
        ),
        Gateway(
            client=admin_client,
            kind_dict=isolation_bootstrap_gateway_body(
                gateway_name=gateway_name,
                gateway_namespace=gateway_namespace,
            ),
            teardown=teardown,
            wait_for_resource=True,
        ) as gateway,
    ):
        verify_maas_gateway_programmed(gateway=gateway)
        LOGGER.info(f"Isolation bootstrap Gateway '{gateway_namespace}/{gateway_name}' is Programmed")
        yield gateway


@contextmanager
def tenant_gateway_external_route(
    admin_client: DynamicClient,
    gateway_name: str,
    maas_host: str,
    teardown: bool,
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
) -> Generator[Route, Any, Any]:
    """Expose a tenant Gateway externally via an edge-terminated OpenShift Route to HTTP port 80."""
    route_host = tenant_gateway_hostname(gateway_name=gateway_name, maas_host=maas_host)
    with Route(
        client=admin_client,
        name=tenant_gateway_route_name(gateway_name=gateway_name),
        namespace=gateway_namespace,
        host=route_host,
        to={"kind": "Service", "name": tenant_gateway_service_name(gateway_name=gateway_name), "weight": 100},
        port={"targetPort": GATEWAY_HTTP_PORT},
        tls={
            "termination": "edge",
            "insecureEdgeTerminationPolicy": "Redirect",
        },
        teardown=teardown,
        wait_for_resource=True,
    ) as route:
        yield route


def extract_api_key_items(list_body: dict[str, Any]) -> list[dict[str, Any]]:
    """Return API key search result items from a maas-api search response body."""
    if "items" in list_body:
        return list_body["items"]
    if "data" in list_body:
        return list_body["data"]
    return []


def assert_api_key_search_excludes_ids(
    list_body: dict[str, Any],
    excluded_key_ids: set[str],
) -> None:
    """Assert the search results do not contain any of the excluded key IDs."""
    found_key_ids = {item["id"] for item in extract_api_key_items(list_body=list_body)}
    overlap = found_key_ids.intersection(excluded_key_ids)
    assert not overlap, f"Search results must not include keys from another tenant, but found ids={sorted(overlap)}"


def assert_api_key_search_includes_ids(
    list_body: dict[str, Any],
    expected_key_ids: set[str],
) -> None:
    """Assert the search results contain all expected key IDs."""
    found_key_ids = {item["id"] for item in extract_api_key_items(list_body=list_body)}
    missing_key_ids = expected_key_ids.difference(found_key_ids)
    assert not missing_key_ids, (
        f"Expected key ids={sorted(expected_key_ids)} in search results, "
        f"missing={sorted(missing_key_ids)}, found={sorted(found_key_ids)}"
    )


@contextmanager
def isolation_tenant_api_key_id(
    request_session_http: requests.Session,
    base_url: str,
    ocp_user_token: str,
    subscription_name: str,
    key_name_prefix: str,
    fixture_label: str,
) -> Generator[str, Any, Any]:
    """Create a tenant-scoped API key, yield its id, then revoke it on teardown."""
    key_name = f"{key_name_prefix}-{generate_random_name()}"
    create_response, api_key_body = create_api_key(
        base_url=base_url,
        ocp_user_token=ocp_user_token,
        request_session_http=request_session_http,
        api_key_name=key_name,
        subscription=subscription_name,
    )
    assert_api_key_created_ok(resp=create_response, body=api_key_body, required_fields=("id",))
    key_id: str = api_key_body["id"]
    LOGGER.info(f"{fixture_label}: created key id={key_id} name={key_name}")
    yield key_id
    revoke_response, _ = revoke_api_key(
        request_session_http=request_session_http,
        base_url=base_url,
        key_id=key_id,
        ocp_user_token=ocp_user_token,
    )
    if revoke_response.status_code not in (200, 404):
        raise AssertionError(
            f"Unexpected teardown status for {fixture_label} key id={key_id}: {revoke_response.status_code}"
        )


def maas_api_deployment_name_for_aitenant(aitenant_name: str) -> str:
    """Return the per-tenant maas-api Deployment name for an additional AITenant."""
    return f"{MAAS_API_DEPLOYMENT_NAME}-{aitenant_name}"


def tenant_gateway_maas_auth_policy_name(gateway_name: str) -> str:
    """Return the MaaS gateway AuthPolicy name reconciled from a tenant MaaSAuthPolicy."""
    return f"{gateway_name}{TENANT_GATEWAY_MAAS_AUTH_POLICY_SUFFIX}"


def maas_api_service_host_for_aitenant(aitenant_name: str, applications_namespace: str) -> str:
    """Return the in-cluster Service hostname for a per-tenant maas-api Deployment."""
    deployment_name = maas_api_deployment_name_for_aitenant(aitenant_name=aitenant_name)
    return f"{deployment_name}.{applications_namespace}.svc.cluster.local"


def shared_maas_api_service_host(applications_namespace: str) -> str:
    """Return the in-cluster Service hostname for the shared maas-api Deployment."""
    return f"{SHARED_MAAS_API_SERVICE_NAME}.{applications_namespace}.svc.cluster.local"


def wait_for_tenant_gateway_maas_auth_policy(
    admin_client: DynamicClient,
    gateway_name: str,
    gateway_namespace: str,
    timeout: int = 300,
) -> AuthPolicy:
    """Poll until the MaaS gateway AuthPolicy for a tenant Gateway exists and is Accepted."""
    policy_name = tenant_gateway_maas_auth_policy_name(gateway_name=gateway_name)
    auth_policy = AuthPolicy(
        client=admin_client,
        name=policy_name,
        namespace=gateway_namespace,
    )
    try:
        for _ in TimeoutSampler(
            wait_timeout=timeout,
            sleep=5,
            func=auth_policy.get,
            exceptions_dict={NotFoundError: [], ResourceNotFoundError: []},
        ):
            if not auth_policy.exists:
                continue
            accepted_condition = get_auth_policy_condition(
                admin_client=admin_client,
                policy_name=policy_name,
                namespace=gateway_namespace,
                condition_type="Accepted",
            )
            if accepted_condition is not None and accepted_condition.get("status") == "True":
                LOGGER.info(
                    f"AuthPolicy '{gateway_namespace}/{policy_name}' is Accepted after MaaSAuthPolicy reconciliation"
                )
                return auth_policy
    except TimeoutExpiredError as error:
        raise AssertionError(
            f"Timed out waiting for AuthPolicy '{gateway_namespace}/{policy_name}' to become Accepted. "
            f"Ensure a MaaSAuthPolicy exists in the tenant namespace to trigger gateway reconciliation."
        ) from error
    raise AssertionError(f"AuthPolicy '{gateway_namespace}/{policy_name}' did not become Accepted")


def verify_tenant_gateway_auth_policy_callback_url(
    admin_client: DynamicClient,
    gateway_name: str,
    gateway_namespace: str,
    aitenant_name: str,
    applications_namespace: str,
) -> str:
    """Assert the tenant Gateway MaaS AuthPolicy callback targets the per-tenant maas-api Service."""
    policy_name = tenant_gateway_maas_auth_policy_name(gateway_name=gateway_name)
    expected_host = maas_api_service_host_for_aitenant(
        aitenant_name=aitenant_name,
        applications_namespace=applications_namespace,
    )
    shared_host = shared_maas_api_service_host(applications_namespace=applications_namespace)

    wait_for_tenant_gateway_maas_auth_policy(
        admin_client=admin_client,
        gateway_name=gateway_name,
        gateway_namespace=gateway_namespace,
    )

    callback_url = get_auth_policy_callback_url(
        admin_client=admin_client,
        policy_name=policy_name,
        namespace=gateway_namespace,
    )
    callback_hostname = urlparse(url=callback_url).hostname
    assert callback_hostname == expected_host, (
        f"AuthPolicy '{gateway_namespace}/{policy_name}' apiKeyValidation callback uses wrong maas-api host. "
        f"Expected hostname '{expected_host}', got '{callback_hostname}' in URL: {callback_url}"
    )
    assert callback_hostname != shared_host, (
        f"AuthPolicy '{gateway_namespace}/{policy_name}' apiKeyValidation callback targets shared maas-api "
        f"('{shared_host}') instead of per-tenant '{expected_host}': {callback_url}"
    )
    LOGGER.info(
        f"AuthPolicy '{gateway_namespace}/{policy_name}' callback correctly targets "
        f"per-tenant maas-api host '{expected_host}': {callback_url}"
    )
    return callback_url


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


def httproute_accepted_on_gateway(
    route: HTTPRoute,
    gateway_name: str,
    gateway_namespace: str,
) -> bool:
    """Return True when the HTTPRoute status reports Accepted=True on the tenant Gateway parent."""
    route_namespace = route.namespace
    parent_statuses = getattr(route.instance.status, "parents", None) or []
    for parent_status in parent_statuses:
        parent_ref = getattr(parent_status, "parentRef", None)
        if parent_ref is None:
            continue
        parent_kind = getattr(parent_ref, "kind", None) or "Gateway"
        if parent_kind != "Gateway":
            continue
        parent_name = getattr(parent_ref, "name", None)
        parent_namespace = getattr(parent_ref, "namespace", None) or route_namespace
        if parent_name != gateway_name or parent_namespace != gateway_namespace:
            continue
        conditions = getattr(parent_status, "conditions", None) or []
        for condition in conditions:
            condition_type = getattr(condition, "type", None)
            condition_status = getattr(condition, "status", None)
            if condition_type == "Accepted" and condition_status == "True":
                return True
    return False


@contextmanager
def patch_llmisvc_with_tenant_gateway_router(
    llm_service: LLMInferenceService,
    gateway_name: str,
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
) -> Generator[None, Any, Any]:
    """Patch an LLMInferenceService to route through a tenant Gateway instead of maas-default-gateway."""
    router_spec = {
        "gateway": {"refs": [{"name": gateway_name, "namespace": gateway_namespace}]},
        "route": {},
    }
    patch_body = {
        "metadata": {
            "annotations": {
                f"alpha.{ApiGroups.MAAS_IO}/tiers": "[]",
                "security.opendatahub.io/enable-auth": "true",
            }
        },
        "spec": {"router": router_spec},
    }
    with ResourceEditor(patches={llm_service: patch_body}):
        yield


@contextmanager
def provision_tenant_model(
    admin_client: DynamicClient,
    model_name: str,
    tenant_namespace_name: str,
    gateway_name: str,
    gateway_namespace: str,
    aws_access_key: str,
    aws_secret_access_key: str,
    aws_s3_region: str,
    aws_s3_bucket: str,
    aws_s3_endpoint: str,
    teardown: bool = True,
    llm_ready_timeout: int = 900,
    httproute_timeout: int = 300,
) -> Generator[MaaSModelRef, Any, Any]:
    """Provision a tenant-local TinyLlama model whose HTTPRoute attaches to the tenant Gateway."""
    label_namespace_gateway_access(
        admin_client=admin_client,
        namespace_name=tenant_namespace_name,
        gateway_name=gateway_name,
    )
    with ExitStack() as stack:
        stack.enter_context(
            cm=s3_endpoint_secret(
                client=admin_client,
                name=TENANT_MODELS_BUCKET_SECRET_NAME,
                namespace=tenant_namespace_name,
                aws_access_key=aws_access_key,
                aws_secret_access_key=aws_secret_access_key,
                aws_s3_region=aws_s3_region,
                aws_s3_bucket=aws_s3_bucket,
                aws_s3_endpoint=aws_s3_endpoint,
                teardown=teardown,
            )
        )
        stack.enter_context(
            cm=ServiceAccount(
                client=admin_client,
                namespace=tenant_namespace_name,
                name=TENANT_MODELS_BUCKET_SA_NAME,
                secrets=[{"name": TENANT_MODELS_BUCKET_SECRET_NAME}],
                teardown=teardown,
            )
        )
        llm_service = stack.enter_context(
            cm=create_llmisvc(
                client=admin_client,
                name=model_name,
                namespace=tenant_namespace_name,
                storage_uri=ModelStorage.S3.TINYLLAMA,
                container_image=ContainerImages.VLLM.CPU,
                container_resources={
                    "limits": {"cpu": "2", "memory": "12Gi"},
                    "requests": {"cpu": "1", "memory": "8Gi"},
                },
                service_account=TENANT_MODELS_BUCKET_SA_NAME,
                wait=False,
                timeout=llm_ready_timeout,
                teardown=teardown,
            )
        )
        stack.enter_context(
            cm=patch_llmisvc_with_tenant_gateway_router(
                llm_service=llm_service,
                gateway_name=gateway_name,
                gateway_namespace=gateway_namespace,
            )
        )
        llm_service.wait_for_condition(
            condition="Ready",
            status="True",
            timeout=llm_ready_timeout,
        )
        wait_for_httproute_accepted_on_gateway(
            admin_client=admin_client,
            route_name=tenant_model_kserve_route_name(model_name=model_name),
            route_namespace=tenant_namespace_name,
            gateway_name=gateway_name,
            gateway_namespace=gateway_namespace,
            timeout=httproute_timeout,
        )
        maas_model = stack.enter_context(
            cm=MaaSModelRef(
                client=admin_client,
                name=model_name,
                namespace=tenant_namespace_name,
                model_ref={
                    "name": model_name,
                    "namespace": tenant_namespace_name,
                    "kind": "LLMInferenceService",
                },
                teardown=teardown,
                wait_for_resource=True,
            )
        )
        LOGGER.info(
            f"Tenant model '{tenant_namespace_name}/{model_name}' is Ready on "
            f"Gateway '{gateway_namespace}/{gateway_name}'"
        )
        yield maas_model


@contextmanager
def make_tenant_model_accessible(
    admin_client: DynamicClient,
    model_name: str,
    model_namespace: str,
    tenant_namespace_name: str,
    auth_policy_name: str,
    subscription_name: str,
    owner_group_name: str,
    free_group_name: str,
    tokens_per_minute: int = 1000,
    window: str = "1m",
    priority: int = 0,
    teardown: bool = True,
    ready_timeout: int = 300,
) -> Generator[MaaSSubscription, Any, Any]:
    """Create tenant MaaSAuthPolicy then MaaSSubscription for a tenant-local model (dev order)."""
    with ExitStack() as stack:
        auth_policy = stack.enter_context(
            cm=MaaSAuthPolicy(
                client=admin_client,
                name=auth_policy_name,
                namespace=tenant_namespace_name,
                model_refs=[{"name": model_name, "namespace": model_namespace}],
                subjects={
                    "groups": [
                        {"name": owner_group_name},
                        {"name": free_group_name},
                    ],
                },
                teardown=teardown,
                wait_for_resource=True,
            )
        )
        auth_policy.wait_for_condition(condition="Ready", status="True", timeout=ready_timeout)
        subscription = stack.enter_context(
            cm=create_maas_subscription(
                admin_client=admin_client,
                subscription_namespace=tenant_namespace_name,
                subscription_name=subscription_name,
                owner_group_name=owner_group_name,
                model_name=model_name,
                model_namespace=model_namespace,
                tokens_per_minute=tokens_per_minute,
                window=window,
                priority=priority,
                teardown=teardown,
                wait_for_resource=True,
            )
        )
        subscription.wait_for_condition(condition="Ready", status="True", timeout=ready_timeout)
        LOGGER.info(
            f"Tenant governance Ready in '{tenant_namespace_name}': "
            f"policy={auth_policy_name}, subscription={subscription_name}"
        )
        yield subscription


def wait_for_httproute_accepted_on_gateway(
    admin_client: DynamicClient,
    route_name: str,
    route_namespace: str,
    gateway_name: str,
    gateway_namespace: str,
    timeout: int = 300,
) -> HTTPRoute:
    """Poll until the per-tenant maas-api HTTPRoute is Accepted by the tenant Gateway."""
    route = HTTPRoute(
        client=admin_client,
        name=route_name,
        namespace=route_namespace,
        ensure_exists=True,
    )
    try:
        for _ in TimeoutSampler(
            wait_timeout=timeout,
            sleep=5,
            func=route.get,
            exceptions_dict={NotFoundError: [], ResourceNotFoundError: []},
        ):
            if httproute_accepted_on_gateway(
                route=route,
                gateway_name=gateway_name,
                gateway_namespace=gateway_namespace,
            ):
                LOGGER.info(
                    f"HTTPRoute/{route_name} in '{route_namespace}' is Accepted by "
                    f"Gateway '{gateway_namespace}/{gateway_name}'"
                )
                return route
    except TimeoutExpiredError:
        parent_conditions = [
            {
                "parentRef": getattr(parent_status, "parentRef", None),
                "conditions": getattr(parent_status, "conditions", None),
            }
            for parent_status in (getattr(route.instance.status, "parents", None) or [])
        ]
        raise AssertionError(
            f"HTTPRoute '{route_namespace}/{route_name}' was not Accepted by "
            f"Gateway '{gateway_namespace}/{gateway_name}' within {timeout}s; "
            f"status.parents={parent_conditions!r}"
        ) from None


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
    wait_for_httproute_accepted_on_gateway(
        admin_client=admin_client,
        route_name=route_name,
        route_namespace=applications_namespace,
        gateway_name=gateway_name,
        gateway_namespace=gateway_namespace,
        timeout=timeout,
    )
    LOGGER.info(
        f"HTTPRoute/{route_name} in '{applications_namespace}' is attached to "
        f"Gateway '{gateway_namespace}/{gateway_name}'"
    )
