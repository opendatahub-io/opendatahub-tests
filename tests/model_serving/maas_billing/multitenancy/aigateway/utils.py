import hashlib
from collections.abc import Callable
from typing import Any, TypedDict

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.gateway_gateway_networking_k8s_io import Gateway
from ocp_resources.namespace import Namespace
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

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
AIGATEWAY_NAME_ANNOTATION = "maas.opendatahub.io/aigateway-name"
AIGATEWAY_NAMESPACE_ANNOTATION = "maas.opendatahub.io/aigateway-namespace"
AIGATEWAY_CREATED_ANNOTATION = "maas.opendatahub.io/created-by-aigateway"
MUTATED_TENANT_NAMESPACE_NAME = "mutated-tenant-ns-maas"
AIGATEWAY_INVALID_PLACEMENT_REASON = "InvalidPlacement"
AIGATEWAY_TENANT_NAMESPACE_MISSING_REASON = "TenantNamespaceMissing"
AIGATEWAY_TENANT_NAMESPACE_FAILED_REASON = "TenantNamespaceFailed"
AIGATEWAY_GATEWAY_RECONCILE_FAILED_REASON = "GatewayReconcileFailed"
AIGATEWAY_CHILD_NAME_PREFIX = "aigateway-"
AIGATEWAY_TENANT_ADMIN_ROLE_SUFFIX = "tenant-admin"
AIGATEWAY_OBJECT_ADMIN_ROLE_SUFFIX = "object-admin"
TEST_RBAC_GROUP_NAME = "maas-aigw-e2e-admins"
AIGATEWAY_TEST_OIDC_SPEC = {
    "issuerUrl": "https://sso.example.com/realms/maas-aigw-e2e",
    "clientId": "maas-aigw-e2e",
    "ttl": 600,
}
AIGATEWAY_TEST_RBAC_ADMINS = [{"kind": "Group", "name": TEST_RBAC_GROUP_NAME}]


class AIGatewayTestContext(TypedDict):
    aigateway: AIGateway
    aigateway_name: str
    tenant_namespace_name: str


class AIGatewayPreexistingNamespaceContext(TypedDict):
    aigateway: AIGateway
    tenant_namespace: Namespace
    tenant_namespace_name: str


def tenant_namespace_name_for_aigateway(aigateway_name: str) -> str:
    """Derive the tenant namespace name created for an AIGateway."""
    return f"{aigateway_name}{AIGATEWAY_TENANT_NAMESPACE_SUFFIX}"


def aigateway_child_resource_name(aigateway_name: str, suffix: str) -> str:
    """Return the controller-derived Role or RoleBinding name for an AIGateway child resource."""
    name = f"{AIGATEWAY_CHILD_NAME_PREFIX}{aigateway_name}-{suffix}"
    if len(name) <= 63:
        return name
    name_hash = hashlib.sha256(aigateway_name.encode()).hexdigest()[:8]
    budget = 63 - len(AIGATEWAY_CHILD_NAME_PREFIX) - len(suffix) - len(name_hash) - 2
    truncated = aigateway_name[:budget] if budget >= 1 else ""
    return f"{AIGATEWAY_CHILD_NAME_PREFIX}{truncated}{name_hash}-{suffix}"


def tenant_admin_role_name(aigateway_name: str) -> str:
    """Return the tenant-admin Role name created for an AIGateway."""
    return aigateway_child_resource_name(
        aigateway_name=aigateway_name,
        suffix=AIGATEWAY_TENANT_ADMIN_ROLE_SUFFIX,
    )


def aigateway_object_admin_role_name(aigateway_name: str) -> str:
    """Return the per-AIGateway access Role name in the infra namespace."""
    return aigateway_child_resource_name(
        aigateway_name=aigateway_name,
        suffix=AIGATEWAY_OBJECT_ADMIN_ROLE_SUFFIX,
    )


def build_aigateway_spec(
    aigateway_name: str,
    tenant_namespace_name: str | None = None,
    cleanup_on_delete: bool = True,
    create_tenant_namespace: bool = True,
    gateway_name: str | None = None,
    gateway_namespace: str | None = None,
    domain: str | None = None,
    tls: dict[str, Any] | None = None,
    oidc: dict[str, Any] | None = None,
    rbac_admins: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build an AIGateway spec for bootstrap and negative-path testing."""
    resolved_tenant_namespace = tenant_namespace_name or tenant_namespace_name_for_aigateway(
        aigateway_name=aigateway_name
    )
    gateway_spec: dict[str, Any] = {
        "namespace": gateway_namespace or MAAS_GATEWAY_NAMESPACE,
        "gatewayClassName": "openshift-default",
    }
    if gateway_name is not None:
        gateway_spec["name"] = gateway_name
    spec: dict[str, Any] = {
        "tenantNamespace": {
            "name": resolved_tenant_namespace,
            "create": create_tenant_namespace,
            "cleanupOnDelete": cleanup_on_delete,
        },
        "gateway": gateway_spec,
    }
    if domain is not None:
        spec["domain"] = domain
    if tls is not None:
        spec["tls"] = tls
    if oidc is not None:
        spec["oidc"] = oidc
    if rbac_admins is not None:
        spec["rbac"] = {"admins": rbac_admins}
    return spec


def aigateway_from_spec(
    admin_client: DynamicClient,
    aigateway_name: str,
    cr_namespace: str,
    aigateway_spec: dict[str, Any],
    teardown: bool = False,
) -> AIGateway:
    """Return an AIGateway configured from spec; use with ``with aigateway_from_spec(...) as aigateway:``."""
    return AIGateway(
        client=admin_client,
        name=aigateway_name,
        namespace=cr_namespace,
        tenant_namespace=aigateway_spec["tenantNamespace"],
        gateway=aigateway_spec["gateway"],
        domain=aigateway_spec.get("domain"),
        tls=aigateway_spec.get("tls"),
        oidc=aigateway_spec.get("oidc"),
        rbac=aigateway_spec.get("rbac"),
        teardown=teardown,
        wait_for_resource=True,
    )


def deploy_and_verify_aigateway_ready(aigateway: AIGateway) -> None:
    """Create the AIGateway CR if missing and wait until it reports Ready with phase Active."""
    if not aigateway.exists:
        aigateway.deploy()
    verify_aigateway_ready(aigateway=aigateway)


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


def verify_bootstrapped_tenant_oidc(
    admin_client: DynamicClient,
    tenant_namespace_name: str,
    expected_oidc: dict[str, Any],
) -> None:
    """Assert bootstrapped Tenant externalOIDC mirrors the AIGateway oidc spec."""
    bootstrapped_tenant = Tenant(
        client=admin_client,
        name=AIGATEWAY_BOOTSTRAPPED_TENANT_NAME,
        namespace=tenant_namespace_name,
        ensure_exists=True,
    )
    tenant_oidc = bootstrapped_tenant.instance.spec.externalOIDC
    assert tenant_oidc is not None, (
        f"Tenant/{AIGATEWAY_BOOTSTRAPPED_TENANT_NAME} in '{tenant_namespace_name}' "
        "should mirror AIGateway oidc into externalOIDC"
    )
    for field_name, expected_value in expected_oidc.items():
        actual_value = getattr(tenant_oidc, field_name, None)
        assert actual_value == expected_value, (
            f"Tenant externalOIDC.{field_name} expected {expected_value!r}, got {actual_value!r}"
        )


def verify_gateway_listener_hostname(
    admin_client: DynamicClient,
    gateway_name: str,
    expected_hostname: str,
) -> None:
    """Assert the tenant Gateway exposes a listener with the expected hostname."""
    tenant_gateway = Gateway(
        client=admin_client,
        name=gateway_name,
        namespace=MAAS_GATEWAY_NAMESPACE,
        ensure_exists=True,
    )
    listeners = tenant_gateway.instance.spec.listeners or []
    assert listeners, f"Gateway '{gateway_name}' has no listeners"
    listener_hostname = listeners[0].hostname
    assert listener_hostname is not None, f"Gateway '{gateway_name}' listener has no hostname"
    assert str(listener_hostname) == expected_hostname, (
        f"Gateway listener hostname expected {expected_hostname!r}, got {listener_hostname!r}"
    )


def verify_gateway_https_listener_tls(
    admin_client: DynamicClient,
    gateway_name: str,
    certificate_secret_name: str,
) -> None:
    """Assert the tenant Gateway exposes an HTTPS listener with the expected TLS cert ref."""
    tenant_gateway = Gateway(
        client=admin_client,
        name=gateway_name,
        namespace=MAAS_GATEWAY_NAMESPACE,
        ensure_exists=True,
    )
    listeners = tenant_gateway.instance.spec.listeners or []
    https_listeners = [listener for listener in listeners if str(getattr(listener, "protocol", "")).endswith("HTTPS")]
    assert https_listeners, f"Gateway '{gateway_name}' has no HTTPS listener"
    tls_config = https_listeners[0].tls
    assert tls_config is not None, f"Gateway '{gateway_name}' HTTPS listener has no TLS config"
    certificate_refs = tls_config.certificateRefs or []
    assert certificate_refs, f"Gateway '{gateway_name}' TLS config has no certificateRefs"
    assert str(certificate_refs[0].name) == certificate_secret_name, (
        f"Expected TLS certificateRef {certificate_secret_name!r}, got {certificate_refs[0].name!r}"
    )


def _normalize_rbac_subjects(subjects: list[Any]) -> list[dict[str, str]]:
    """Return RoleBinding subjects as kind/name pairs for assertions."""
    return [{"kind": subject.kind, "name": subject.name} for subject in subjects]


def verify_aigateway_role_binding(
    admin_client: DynamicClient,
    namespace: str,
    binding_name: str,
    role_name: str,
    expected_subjects: list[dict[str, str]] | None = None,
    should_exist: bool = True,
) -> None:
    """Assert a namespaced RoleBinding exists with the expected roleRef and optional subjects."""
    role_binding = RoleBinding(
        client=admin_client,
        name=binding_name,
        namespace=namespace,
        ensure_exists=should_exist,
    )
    if not should_exist:
        assert not role_binding.exists, f"RoleBinding '{namespace}/{binding_name}' should not exist"
        return
    assert role_binding.exists, f"RoleBinding '{namespace}/{binding_name}' was not created"
    assert role_binding.instance.roleRef.kind == "Role", (
        f"RoleBinding '{binding_name}' roleRef.kind expected Role, got {role_binding.instance.roleRef.kind!r}"
    )
    assert role_binding.instance.roleRef.name == role_name, (
        f"RoleBinding '{binding_name}' roleRef.name expected {role_name!r}, got {role_binding.instance.roleRef.name!r}"
    )
    if expected_subjects is not None:
        actual_subjects = _normalize_rbac_subjects(subjects=role_binding.instance.subjects or [])
        assert actual_subjects == expected_subjects, (
            f"RoleBinding '{namespace}/{binding_name}' subjects expected {expected_subjects!r}, got {actual_subjects!r}"
        )


def verify_aigateway_rbac_admins_bindings(
    admin_client: DynamicClient,
    aigateway_name: str,
    tenant_namespace_name: str,
    infra_namespace: str,
    expected_admins: list[dict[str, str]],
) -> None:
    """Assert tenant-admin and object-admin RoleBindings exist with spec.rbac.admins subjects."""
    tenant_admin_name = tenant_admin_role_name(aigateway_name=aigateway_name)
    object_admin_name = aigateway_object_admin_role_name(aigateway_name=aigateway_name)
    verify_aigateway_role_binding(
        admin_client=admin_client,
        namespace=tenant_namespace_name,
        binding_name=tenant_admin_name,
        role_name=tenant_admin_name,
        expected_subjects=expected_admins,
    )
    verify_aigateway_role_binding(
        admin_client=admin_client,
        namespace=infra_namespace,
        binding_name=object_admin_name,
        role_name=object_admin_name,
        expected_subjects=expected_admins,
    )


def verify_aigateway_rbac_roles_without_admin_bindings(
    admin_client: DynamicClient,
    aigateway_name: str,
    tenant_namespace_name: str,
    infra_namespace: str,
) -> None:
    """Assert Roles exist but admin RoleBindings are omitted when spec.rbac.admins is unset."""
    tenant_admin_name = tenant_admin_role_name(aigateway_name=aigateway_name)
    object_admin_name = aigateway_object_admin_role_name(aigateway_name=aigateway_name)
    tenant_role = Role(client=admin_client, name=tenant_admin_name, namespace=tenant_namespace_name)
    infra_role = Role(client=admin_client, name=object_admin_name, namespace=infra_namespace)
    assert tenant_role.exists, f"Role '{tenant_namespace_name}/{tenant_admin_name}' should exist"
    assert infra_role.exists, f"Role '{infra_namespace}/{object_admin_name}' should exist"
    verify_aigateway_role_binding(
        admin_client=admin_client,
        namespace=tenant_namespace_name,
        binding_name=tenant_admin_name,
        role_name=tenant_admin_name,
        should_exist=False,
    )
    verify_aigateway_role_binding(
        admin_client=admin_client,
        namespace=infra_namespace,
        binding_name=object_admin_name,
        role_name=object_admin_name,
        should_exist=False,
    )


def _wait_until_resource_absent(
    *,
    exists_check: Callable[[], bool],
    resource_label: str,
    timeout: int = 300,
) -> None:
    """Poll until exists_check() returns False (resource deleted from the API)."""
    try:
        for absent in TimeoutSampler(
            wait_timeout=timeout,
            sleep=5,
            func=lambda: not exists_check(),
        ):
            if absent:
                return
    except TimeoutExpiredError:
        pytest.fail(f"{resource_label} still exists after AIGateway deletion (timeout {timeout}s)")


def verify_aigateway_bootstrap_children_removed(
    admin_client: DynamicClient,
    test_context: AIGatewayTestContext,
    timeout: int = 300,
) -> None:
    """Assert tenant namespace, Gateway, and bootstrapped Tenant were removed."""
    aigateway_name = test_context["aigateway_name"]
    tenant_namespace_name = test_context["tenant_namespace_name"]

    _wait_until_resource_absent(
        exists_check=lambda: Namespace(client=admin_client, name=tenant_namespace_name).exists,
        resource_label=f"Tenant namespace '{tenant_namespace_name}'",
        timeout=timeout,
    )

    _wait_until_resource_absent(
        exists_check=lambda: (
            Gateway(
                client=admin_client,
                name=aigateway_name,
                namespace=MAAS_GATEWAY_NAMESPACE,
            ).exists
        ),
        resource_label=f"Gateway '{aigateway_name}' in '{MAAS_GATEWAY_NAMESPACE}'",
        timeout=timeout,
    )

    _wait_until_resource_absent(
        exists_check=lambda: (
            Tenant(
                client=admin_client,
                name=AIGATEWAY_BOOTSTRAPPED_TENANT_NAME,
                namespace=tenant_namespace_name,
            ).exists
        ),
        resource_label=(f"Tenant/{AIGATEWAY_BOOTSTRAPPED_TENANT_NAME} in '{tenant_namespace_name}'"),
        timeout=timeout,
    )


def verify_tenant_namespace_preserved(
    admin_client: DynamicClient,
    tenant_namespace_name: str,
) -> None:
    """Assert the tenant namespace still exists after AIGateway deletion."""
    tenant_namespace = Namespace(
        client=admin_client,
        name=tenant_namespace_name,
        ensure_exists=True,
    )
    assert tenant_namespace.exists, (
        f"Tenant namespace '{tenant_namespace_name}' should be preserved when cleanupOnDelete=false"
    )


def _fresh_aigateway(aigateway: AIGateway) -> AIGateway:
    """Return a new handle to re-read the current AIGateway status from the API."""
    return AIGateway(
        client=aigateway.client,
        name=aigateway.name,
        namespace=aigateway.namespace,
        wait_for_resource=False,
    )


def get_aigateway_ready_reason(aigateway: AIGateway) -> str:
    """Return the Ready condition reason, or an empty string when absent."""
    fresh_aigateway = _fresh_aigateway(aigateway=aigateway)
    for condition in fresh_aigateway.instance.status.conditions or []:
        if condition.type == "Ready":
            return condition.reason or ""
    return ""


def aigateway_has_status(
    aigateway: AIGateway,
    phase: str,
    ready_reason: str | None = None,
) -> bool:
    """Return True when AIGateway status matches the expected phase and optional Ready reason."""
    fresh_aigateway = _fresh_aigateway(aigateway=aigateway)
    current_phase = getattr(fresh_aigateway.instance.status, "phase", "") or ""
    if current_phase != phase:
        return False
    if ready_reason is None:
        return True
    return get_aigateway_ready_reason(aigateway=aigateway) == ready_reason


def wait_until_aigateway_status(
    aigateway: AIGateway,
    phase: str,
    ready_reason: str | None = None,
    timeout: int = 120,
) -> None:
    """Wait until AIGateway reaches the expected phase and optional Ready reason."""
    try:
        for matched in TimeoutSampler(
            wait_timeout=timeout,
            sleep=5,
            func=lambda: aigateway_has_status(
                aigateway=aigateway,
                phase=phase,
                ready_reason=ready_reason,
            ),
        ):
            if matched:
                return
    except TimeoutExpiredError:
        current_phase = getattr(_fresh_aigateway(aigateway=aigateway).instance.status, "phase", "") or ""
        current_reason = get_aigateway_ready_reason(aigateway=aigateway)
        pytest.fail(
            f"AIGateway '{aigateway.name}' did not reach phase={phase} "
            f"ready_reason={ready_reason}: phase={current_phase} ready_reason={current_reason}"
        )


def verify_aigateway_invalid_placement(aigateway: AIGateway) -> None:
    """Assert the controller rejected AIGateway placement with InvalidPlacement."""
    wait_until_aigateway_status(
        aigateway=aigateway,
        phase="Failed",
        ready_reason=AIGATEWAY_INVALID_PLACEMENT_REASON,
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
