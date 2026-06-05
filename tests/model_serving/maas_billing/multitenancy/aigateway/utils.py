import hashlib
from collections.abc import Callable, Generator
from contextlib import contextmanager
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
from utilities.resources.aitenant import AITenant
from utilities.resources.tenant import Tenant

LOGGER = structlog.get_logger(name=__name__)

AIGATEWAY_CRD_NAME = f"aitenants.{ApiGroups.MAAS_IO}"
AIGATEWAY_INFRA_NAMESPACE = "redhat-ai-gateway-infra"
AIGATEWAY_BOOTSTRAPPED_TENANT_NAME = "default-tenant"
AIGATEWAY_TENANT_NAMESPACE_SUFFIX = "-maas"
AIGATEWAY_NAME_ANNOTATION = "maas.opendatahub.io/aitenant-name"
AIGATEWAY_NAMESPACE_ANNOTATION = "maas.opendatahub.io/aitenant-namespace"
AIGATEWAY_CREATED_ANNOTATION = "maas.opendatahub.io/created-by-aitenant"
MUTATED_TENANT_NAMESPACE_NAME = "mutated-tenant-ns-maas"
AIGATEWAY_INVALID_PLACEMENT_REASON = "InvalidPlacement"
AIGATEWAY_TENANT_NAMESPACE_MISSING_REASON = "TenantNamespaceMissing"
AIGATEWAY_TENANT_NAMESPACE_FAILED_REASON = "TenantNamespaceFailed"
AIGATEWAY_CHILD_NAME_PREFIX = "aitenant-"
AIGATEWAY_TENANT_ADMIN_ROLE_SUFFIX = "tenant-admin"
AIGATEWAY_OBJECT_ADMIN_ROLE_SUFFIX = "object-admin"
TEST_RBAC_GROUP_NAME = "maas-aigw-e2e-admins"
AIGATEWAY_TEST_OIDC_SPEC = {
    "issuerUrl": "https://sso.example.com/realms/maas-aigw-e2e",
    "clientId": "maas-aigw-e2e",
    "ttl": 600,
}
AIGATEWAY_TEST_RBAC_ADMINS = [{"kind": "Group", "name": TEST_RBAC_GROUP_NAME}]
AIGATEWAY_GATEWAY_CLASS_NAME = "openshift-default"
AIGATEWAY_BOOTSTRAP_GATEWAY_LISTENERS = [{"name": "http", "port": 80, "protocol": "HTTP"}]
AIGATEWAY_MANAGED_BY_LABEL = "maas.opendatahub.io/managed-by-aitenant"
AIGATEWAY_TENANT_LABEL = "ai-gateway.opendatahub.io/tenant"


class AIGatewayTestContext(TypedDict):
    aigateway: AITenant
    aigateway_name: str
    tenant_namespace_name: str


class AIGatewayPreexistingNamespaceContext(TypedDict):
    aigateway: AITenant
    tenant_namespace: Namespace
    tenant_namespace_name: str


def tenant_namespace_name_for_aigateway(aigateway_name: str) -> str:
    """Derive the tenant namespace name created for an AIGateway."""
    return f"{aigateway_name}{AIGATEWAY_TENANT_NAMESPACE_SUFFIX}"


def tenant_namespace_name_from_aigateway(aigateway: AITenant) -> str:
    """Return the tenant namespace name configured on a deployed AITenant."""
    if aigateway.tenant_namespace is not None:
        configured_name = aigateway.tenant_namespace.get("name")
        if configured_name:
            return configured_name
    return aigateway.instance.spec.tenantNamespace.name


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
    include_gateway: bool = False,
    oidc: dict[str, Any] | None = None,
    rbac_admins: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build an AITenant spec for bootstrap and negative-path testing."""
    resolved_tenant_namespace = tenant_namespace_name or tenant_namespace_name_for_aigateway(
        aigateway_name=aigateway_name
    )
    spec: dict[str, Any] = {
        "tenantNamespace": {
            "name": resolved_tenant_namespace,
            "create": create_tenant_namespace,
            "cleanupOnDelete": cleanup_on_delete,
        },
    }
    if include_gateway:
        gateway_spec: dict[str, Any] = {
            "namespace": gateway_namespace or MAAS_GATEWAY_NAMESPACE,
            "gatewayClassName": "openshift-default",
        }
        if gateway_name is not None:
            gateway_spec["name"] = gateway_name
        spec["gateway"] = gateway_spec
    if oidc is not None:
        spec["oidc"] = oidc
    if rbac_admins is not None:
        spec["rbac"] = {"admins": rbac_admins}
    return spec


def bootstrap_gateway_ref(
    aigateway_name: str,
    aigateway_spec: dict[str, Any],
) -> tuple[str, str]:
    """Resolve the bootstrap Gateway name and namespace from an AITenant spec."""
    gateway_spec = aigateway_spec.get("gateway", {})
    return (
        gateway_spec.get("name", aigateway_name),
        gateway_spec.get("namespace", MAAS_GATEWAY_NAMESPACE),
    )


def bootstrap_gateway_ref_from_aigateway(aigateway: AITenant) -> tuple[str, str]:
    """Resolve the bootstrap Gateway name and namespace from an AITenant configuration."""
    aigateway_spec: dict[str, Any] = {}
    if aigateway.gateway is not None:
        aigateway_spec["gateway"] = aigateway.gateway
    return bootstrap_gateway_ref(
        aigateway_name=aigateway.name,
        aigateway_spec=aigateway_spec,
    )


def aigateway_from_spec(
    admin_client: DynamicClient,
    aigateway_name: str,
    cr_namespace: str,
    aigateway_spec: dict[str, Any],
    teardown: bool = False,
) -> AITenant:
    """Return an AITenant configured from spec; use with ``with aigateway_from_spec(...) as aigateway:``."""
    aitenant_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": aigateway_name,
        "namespace": cr_namespace,
        "tenant_namespace": aigateway_spec["tenantNamespace"],
        "teardown": teardown,
        "wait_for_resource": True,
    }
    if "gateway" in aigateway_spec:
        aitenant_kwargs["gateway"] = aigateway_spec["gateway"]
    if "oidc" in aigateway_spec:
        aitenant_kwargs["oidc"] = aigateway_spec["oidc"]
    if "rbac" in aigateway_spec:
        aitenant_kwargs["rbac"] = aigateway_spec["rbac"]
    return AITenant(**aitenant_kwargs)


def aigateway_bootstrap_gateway(
    admin_client: DynamicClient,
    gateway_name: str,
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
    teardown: bool = True,
) -> Gateway:
    """Return a bootstrap Gateway that must exist before AITenant reconciliation."""
    return Gateway(
        client=admin_client,
        name=gateway_name,
        namespace=gateway_namespace,
        gateway_class_name=AIGATEWAY_GATEWAY_CLASS_NAME,
        listeners=AIGATEWAY_BOOTSTRAP_GATEWAY_LISTENERS,
        teardown=teardown,
    )


@contextmanager
def ready_aigateway_with_preprovisioned_gateway(
    admin_client: DynamicClient,
    aigateway_name: str,
    cr_namespace: str,
    aigateway_spec: dict[str, Any],
    teardown: bool,
) -> Generator[AITenant]:
    """Pre-provision the bootstrap Gateway, deploy AITenant, and wait until Ready."""
    gateway_name, gateway_namespace = bootstrap_gateway_ref(
        aigateway_name=aigateway_name,
        aigateway_spec=aigateway_spec,
    )
    with (
        aigateway_bootstrap_gateway(
            admin_client=admin_client,
            gateway_name=gateway_name,
            gateway_namespace=gateway_namespace,
            teardown=teardown,
        ),
        aigateway_from_spec(
            admin_client=admin_client,
            aigateway_name=aigateway_name,
            cr_namespace=cr_namespace,
            aigateway_spec=aigateway_spec,
            teardown=teardown,
        ) as aigateway,
    ):
        deploy_and_verify_aigateway_ready(aigateway=aigateway)
        yield aigateway


def deploy_and_verify_aigateway_ready(aigateway: AITenant) -> None:
    """Create the AITenant CR if missing and wait until it reports Ready with phase Active."""
    if not aigateway.exists:
        aigateway.deploy()
    verify_aigateway_ready(aigateway=aigateway)


def build_aigateway_test_context(aigateway: AITenant) -> AIGatewayTestContext:
    """Build the standard test context dict from a deployed AITenant."""
    return AIGatewayTestContext(
        aigateway=aigateway,
        aigateway_name=aigateway.name,
        tenant_namespace_name=tenant_namespace_name_from_aigateway(aigateway=aigateway),
    )


def verify_aigateway_ready(aigateway: AITenant) -> None:
    """Assert the AITenant exists and reports Ready=True with phase Active."""
    assert aigateway.exists, f"AITenant '{aigateway.name}' not found in namespace '{aigateway.namespace}'"
    aigateway.wait_for_condition(condition="Ready", status="True", timeout=300)
    phase = getattr(aigateway.instance.status, "phase", "") or ""
    assert phase == "Active", f"Expected AITenant phase Active, got '{phase}'"


def verify_aigateway_bootstrap_children(
    admin_client: DynamicClient,
    test_context: AIGatewayTestContext,
    infra_namespace: str = AIGATEWAY_INFRA_NAMESPACE,
) -> None:
    """Assert AITenant bootstrap created the expected namespace, Gateway, and Tenant resources."""
    aigateway = test_context["aigateway"]
    aigateway_name = test_context["aigateway_name"]
    tenant_namespace_name = test_context["tenant_namespace_name"]

    fresh_aigateway = _fresh_aigateway(aigateway=aigateway)
    aigateway_status = fresh_aigateway.instance.status
    status_gateway_ref = getattr(aigateway_status, "gatewayRef", None)
    assert status_gateway_ref is not None, (
        f"AITenant '{aigateway_name}' status.gatewayRef should be set after bootstrap"
    )
    gateway_name = status_gateway_ref.name
    gateway_namespace = status_gateway_ref.namespace
    status_tenant_namespace = getattr(aigateway_status, "tenantNamespace", None)
    assert status_tenant_namespace == tenant_namespace_name, (
        f"AITenant status.tenantNamespace expected {tenant_namespace_name!r}, got {status_tenant_namespace!r}"
    )

    tenant_namespace = Namespace(
        client=admin_client,
        name=tenant_namespace_name,
        ensure_exists=True,
    )
    assert tenant_namespace.exists, f"Tenant namespace '{tenant_namespace_name}' was not created"
    namespace_labels = dict(tenant_namespace.instance.metadata.labels or {})
    namespace_annotations = dict(tenant_namespace.instance.metadata.annotations or {})
    assert namespace_labels.get(AIGATEWAY_MANAGED_BY_LABEL) == "true", (
        f"Tenant namespace '{tenant_namespace_name}' label {AIGATEWAY_MANAGED_BY_LABEL} expected 'true', "
        f"got {namespace_labels.get(AIGATEWAY_MANAGED_BY_LABEL)!r}"
    )
    assert namespace_labels.get(AIGATEWAY_TENANT_LABEL) == aigateway_name, (
        f"Tenant namespace '{tenant_namespace_name}' label {AIGATEWAY_TENANT_LABEL} expected {aigateway_name!r}, "
        f"got {namespace_labels.get(AIGATEWAY_TENANT_LABEL)!r}"
    )
    assert namespace_annotations.get(AIGATEWAY_NAME_ANNOTATION) == aigateway_name, (
        f"Tenant namespace {AIGATEWAY_NAME_ANNOTATION} expected {aigateway_name!r}, "
        f"got {namespace_annotations.get(AIGATEWAY_NAME_ANNOTATION)!r}"
    )
    assert namespace_annotations.get(AIGATEWAY_NAMESPACE_ANNOTATION) == infra_namespace, (
        f"Tenant namespace {AIGATEWAY_NAMESPACE_ANNOTATION} expected {infra_namespace!r}, "
        f"got {namespace_annotations.get(AIGATEWAY_NAMESPACE_ANNOTATION)!r}"
    )

    tenant_gateway = Gateway(
        client=admin_client,
        name=gateway_name,
        namespace=gateway_namespace,
        ensure_exists=True,
    )
    gateway_labels = dict(tenant_gateway.instance.metadata.labels or {})
    gateway_annotations = dict(tenant_gateway.instance.metadata.annotations or {})
    for metadata_name, metadata in (
        ("labels", gateway_labels),
        ("annotations", gateway_annotations),
    ):
        assert AIGATEWAY_NAME_ANNOTATION not in metadata, (
            f"Pre-provisioned Gateway '{gateway_namespace}/{gateway_name}' should not have "
            f"{metadata_name} {AIGATEWAY_NAME_ANNOTATION!r}"
        )
        assert AIGATEWAY_NAMESPACE_ANNOTATION not in metadata, (
            f"Pre-provisioned Gateway '{gateway_namespace}/{gateway_name}' should not have "
            f"{metadata_name} {AIGATEWAY_NAMESPACE_ANNOTATION!r}"
        )
    gateway_class_name = getattr(tenant_gateway.instance.spec, "gatewayClassName", None)
    assert gateway_class_name == AIGATEWAY_GATEWAY_CLASS_NAME, (
        f"Gateway '{gateway_namespace}/{gateway_name}' gatewayClassName expected "
        f"{AIGATEWAY_GATEWAY_CLASS_NAME!r}, got {gateway_class_name!r}"
    )
    verify_maas_gateway_programmed(gateway=tenant_gateway)

    bootstrapped_tenant = Tenant(
        client=admin_client,
        name=AIGATEWAY_BOOTSTRAPPED_TENANT_NAME,
        namespace=tenant_namespace_name,
        ensure_exists=True,
    )
    assert bootstrapped_tenant.exists, (
        f"Tenant/{AIGATEWAY_BOOTSTRAPPED_TENANT_NAME} was not created in '{tenant_namespace_name}'"
    )
    tenant_labels = dict(bootstrapped_tenant.instance.metadata.labels or {})
    assert tenant_labels.get(AIGATEWAY_MANAGED_BY_LABEL) is not None, (
        f"Tenant/{AIGATEWAY_BOOTSTRAPPED_TENANT_NAME} should have label {AIGATEWAY_MANAGED_BY_LABEL}"
    )
    tenant_gateway_ref = getattr(bootstrapped_tenant.instance.spec, "gatewayRef", None)
    assert tenant_gateway_ref is not None, (
        f"Tenant/{AIGATEWAY_BOOTSTRAPPED_TENANT_NAME} spec.gatewayRef should be set after bootstrap"
    )
    assert tenant_gateway_ref.name == gateway_name, (
        f"Tenant gatewayRef.name expected {gateway_name!r}, got {tenant_gateway_ref.name!r}"
    )
    assert tenant_gateway_ref.namespace == gateway_namespace, (
        f"Tenant gatewayRef.namespace expected {gateway_namespace!r}, got {tenant_gateway_ref.namespace!r}"
    )
    LOGGER.info(
        f"AIGateway '{aigateway_name}' bootstrap verified: namespace, gateway, and "
        f"Tenant/{AIGATEWAY_BOOTSTRAPPED_TENANT_NAME} exist with expected metadata"
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
        pytest.fail(f"{resource_label} still exists after AITenant deletion (timeout {timeout}s)")


def verify_aigateway_rbac_children_removed(
    admin_client: DynamicClient,
    aigateway_name: str,
    tenant_namespace_name: str,
    infra_namespace: str,
    timeout: int = 300,
) -> None:
    """Assert tenant-admin and object-admin Roles and RoleBindings were removed after AITenant deletion."""
    tenant_admin_name = tenant_admin_role_name(aigateway_name=aigateway_name)
    object_admin_name = aigateway_object_admin_role_name(aigateway_name=aigateway_name)
    _wait_until_resource_absent(
        exists_check=lambda: (
            Role(
                client=admin_client,
                name=tenant_admin_name,
                namespace=tenant_namespace_name,
            ).exists
        ),
        resource_label=f"Role '{tenant_namespace_name}/{tenant_admin_name}'",
        timeout=timeout,
    )
    _wait_until_resource_absent(
        exists_check=lambda: (
            RoleBinding(
                client=admin_client,
                name=tenant_admin_name,
                namespace=tenant_namespace_name,
            ).exists
        ),
        resource_label=f"RoleBinding '{tenant_namespace_name}/{tenant_admin_name}'",
        timeout=timeout,
    )
    _wait_until_resource_absent(
        exists_check=lambda: (
            Role(
                client=admin_client,
                name=object_admin_name,
                namespace=infra_namespace,
            ).exists
        ),
        resource_label=f"Role '{infra_namespace}/{object_admin_name}'",
        timeout=timeout,
    )
    _wait_until_resource_absent(
        exists_check=lambda: (
            RoleBinding(
                client=admin_client,
                name=object_admin_name,
                namespace=infra_namespace,
            ).exists
        ),
        resource_label=f"RoleBinding '{infra_namespace}/{object_admin_name}'",
        timeout=timeout,
    )


def verify_preprovisioned_bootstrap_gateway_preserved(
    admin_client: DynamicClient,
    gateway_name: str,
    gateway_namespace: str,
) -> None:
    """Assert the pre-provisioned bootstrap Gateway still exists after AITenant deletion."""
    bootstrap_gateway = Gateway(
        client=admin_client,
        name=gateway_name,
        namespace=gateway_namespace,
    )
    assert bootstrap_gateway.exists, (
        f"Pre-provisioned Gateway '{gateway_namespace}/{gateway_name}' should be preserved after AITenant deletion"
    )


def verify_tenant_namespace_aitenant_metadata_stripped(
    admin_client: DynamicClient,
    tenant_namespace_name: str,
) -> None:
    """Assert AITenant ownership labels and annotations were removed from the tenant namespace."""
    tenant_namespace = Namespace(
        client=admin_client,
        name=tenant_namespace_name,
        ensure_exists=True,
    )
    labels = tenant_namespace.instance.metadata.labels or {}
    annotations = tenant_namespace.instance.metadata.annotations or {}
    assert labels.get(AIGATEWAY_MANAGED_BY_LABEL) is None, (
        f"Tenant namespace '{tenant_namespace_name}' should not retain {AIGATEWAY_MANAGED_BY_LABEL}"
    )
    assert labels.get(AIGATEWAY_TENANT_LABEL) is None, (
        f"Tenant namespace '{tenant_namespace_name}' should not retain {AIGATEWAY_TENANT_LABEL}"
    )
    assert annotations.get(AIGATEWAY_NAME_ANNOTATION) is None, (
        f"Tenant namespace '{tenant_namespace_name}' should not retain {AIGATEWAY_NAME_ANNOTATION}"
    )
    assert annotations.get(AIGATEWAY_NAMESPACE_ANNOTATION) is None, (
        f"Tenant namespace '{tenant_namespace_name}' should not retain {AIGATEWAY_NAMESPACE_ANNOTATION}"
    )


def verify_aigateway_bootstrap_children_removed(
    admin_client: DynamicClient,
    test_context: AIGatewayTestContext,
    infra_namespace: str = AIGATEWAY_INFRA_NAMESPACE,
    timeout: int = 300,
) -> None:
    """Assert controller-owned Tenant and RBAC children were removed after AITenant deletion."""
    aigateway = test_context["aigateway"]
    aigateway_name = test_context["aigateway_name"]
    tenant_namespace_name = test_context["tenant_namespace_name"]
    gateway_name, gateway_namespace = bootstrap_gateway_ref_from_aigateway(aigateway=aigateway)

    verify_preprovisioned_bootstrap_gateway_preserved(
        admin_client=admin_client,
        gateway_name=gateway_name,
        gateway_namespace=gateway_namespace,
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

    verify_aigateway_rbac_children_removed(
        admin_client=admin_client,
        aigateway_name=aigateway_name,
        tenant_namespace_name=tenant_namespace_name,
        infra_namespace=infra_namespace,
        timeout=timeout,
    )


def verify_tenant_namespace_preserved(
    admin_client: DynamicClient,
    tenant_namespace_name: str,
) -> None:
    """Assert the tenant namespace still exists after AITenant deletion."""
    tenant_namespace = Namespace(
        client=admin_client,
        name=tenant_namespace_name,
        ensure_exists=True,
    )
    assert tenant_namespace.exists, (
        f"Tenant namespace '{tenant_namespace_name}' should be preserved after AITenant deletion"
    )


def _fresh_aigateway(aigateway: AITenant) -> AITenant:
    """Return a new handle to re-read the current AITenant status from the API."""
    return AITenant(
        client=aigateway.client,
        name=aigateway.name,
        namespace=aigateway.namespace,
        wait_for_resource=False,
    )


def get_aigateway_ready_reason(aigateway: AITenant) -> str:
    """Return the Ready condition reason, or an empty string when absent."""
    fresh_aigateway = _fresh_aigateway(aigateway=aigateway)
    for condition in fresh_aigateway.instance.status.conditions or []:
        if condition.type == "Ready":
            return condition.reason or ""
    return ""


def aigateway_has_status(
    aigateway: AITenant,
    phase: str,
    ready_reason: str | None = None,
) -> bool:
    """Return True when AITenant status matches the expected phase and optional Ready reason."""
    fresh_aigateway = _fresh_aigateway(aigateway=aigateway)
    current_phase = getattr(fresh_aigateway.instance.status, "phase", "") or ""
    if current_phase != phase:
        return False
    if ready_reason is None:
        return True
    return get_aigateway_ready_reason(aigateway=aigateway) == ready_reason


def wait_until_aigateway_status(
    aigateway: AITenant,
    phase: str,
    ready_reason: str | None = None,
    timeout: int = 120,
) -> None:
    """Wait until AITenant reaches the expected phase and optional Ready reason."""
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
            f"AITenant '{aigateway.name}' did not reach phase={phase} "
            f"ready_reason={ready_reason}: phase={current_phase} ready_reason={current_reason}"
        )


def verify_aigateway_invalid_placement(aigateway: AITenant) -> None:
    """Assert the controller rejected AITenant placement with InvalidPlacement."""
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
