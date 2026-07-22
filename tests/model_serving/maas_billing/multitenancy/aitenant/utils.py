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
from tests.model_serving.maas_billing.utils import (
    verify_maas_gateway_programmed,
    verify_maas_tenant_config_ready,
)
from utilities.constants import MAAS_GATEWAY_NAMESPACE, ApiGroups
from utilities.resources.aitenant import AITenant
from utilities.resources.maastenantconfig import MaasTenantConfig
from utilities.resources.tenant import Tenant

LOGGER = structlog.get_logger(name=__name__)

AITENANT_CRD_NAME = f"aitenants.{ApiGroups.MAAS_IO}"
AITENANT_INFRA_NAMESPACE = "ai-tenants"
AITENANT_TENANT_NAMESPACE_PREFIX = "ai-tenant-"
AIGATEWAY_BOOTSTRAPPED_TENANT_NAME = "default-tenant"
AIGATEWAY_NAME_ANNOTATION = "maas.opendatahub.io/aitenant-name"
AIGATEWAY_NAMESPACE_ANNOTATION = "maas.opendatahub.io/aitenant-namespace"
AIGATEWAY_CREATED_ANNOTATION = "maas.opendatahub.io/created-by-aitenant"
AITENANT_TENANT_NAMESPACE_FAILED_REASON = "TenantNamespaceFailed"
AIGATEWAY_CHILD_NAME_PREFIX = "aitenant-"
AIGATEWAY_TENANT_ADMIN_ROLE_SUFFIX = "tenant-admin"
AIGATEWAY_OBJECT_ADMIN_ROLE_SUFFIX = "object-admin"
TEST_RBAC_GROUP_NAME = "maas-aigw-e2e-admins"
AITENANT_TEST_OIDC_SPEC = {
    "issuerUrl": "https://sso.example.com/realms/maas-aigw-e2e",
    "clientId": "maas-aigw-e2e",
    "ttl": 600,
}
AITENANT_TEST_RBAC_ADMINS = [{"kind": "Group", "name": TEST_RBAC_GROUP_NAME}]
AIGATEWAY_GATEWAY_CLASS_NAME = "openshift-default"
AIGATEWAY_BOOTSTRAP_GATEWAY_LISTENERS = [{"name": "http", "port": 80, "protocol": "HTTP"}]
AIGATEWAY_MANAGED_BY_LABEL = "maas.opendatahub.io/managed-by-aitenant"
AIGATEWAY_TENANT_LABEL = "ai-gateway.opendatahub.io/tenant"


class AITenantTestContext(TypedDict):
    aitenant: AITenant
    aitenant_name: str
    tenant_namespace_name: str


class AITenantPreexistingNamespaceContext(TypedDict):
    aitenant: AITenant
    tenant_namespace: Namespace
    tenant_namespace_name: str


def expected_tenant_namespace_name(aitenant_name: str) -> str:
    """Return the tenant namespace name the controller derives for an AITenant."""
    if aitenant_name == MAAS_SUBSCRIPTION_NAMESPACE:
        return MAAS_SUBSCRIPTION_NAMESPACE
    return f"{AITENANT_TENANT_NAMESPACE_PREFIX}{aitenant_name}"


def tenant_namespace_name_from_aitenant(aitenant: AITenant) -> str:
    """Return the reconciled tenant namespace name from AITenant status."""
    fresh_aitenant = _fresh_aitenant(aitenant=aitenant)
    status_tenant_namespace = getattr(fresh_aitenant.instance.status, "tenantNamespace", None)
    if status_tenant_namespace:
        return status_tenant_namespace
    return expected_tenant_namespace_name(aitenant_name=aitenant.name)


def aitenant_child_resource_name(aitenant_name: str, suffix: str) -> str:
    """Return the controller-derived Role or RoleBinding name for an AITenant child resource."""
    name = f"{AIGATEWAY_CHILD_NAME_PREFIX}{aitenant_name}-{suffix}"
    if len(name) <= 63:
        return name
    name_hash = hashlib.sha256(aitenant_name.encode()).hexdigest()[:8]
    budget = 63 - len(AIGATEWAY_CHILD_NAME_PREFIX) - len(suffix) - len(name_hash) - 2
    truncated = aitenant_name[:budget] if budget >= 1 else ""
    return f"{AIGATEWAY_CHILD_NAME_PREFIX}{truncated}{name_hash}-{suffix}"


def tenant_admin_role_name(aitenant_name: str) -> str:
    """Return the tenant-admin Role name created for an AITenant."""
    return aitenant_child_resource_name(
        aitenant_name=aitenant_name,
        suffix=AIGATEWAY_TENANT_ADMIN_ROLE_SUFFIX,
    )


def aitenant_object_admin_role_name(aitenant_name: str) -> str:
    """Return the per-AITenant access Role name in the infra namespace."""
    return aitenant_child_resource_name(
        aitenant_name=aitenant_name,
        suffix=AIGATEWAY_OBJECT_ADMIN_ROLE_SUFFIX,
    )


def build_aitenant_spec(
    aitenant_name: str,
    gateway_name: str | None = None,
    oidc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an AITenant spec with gateway and optional oidc fields."""
    spec: dict[str, Any] = {}
    resolved_gateway_name = gateway_name or aitenant_name
    spec["gateway"] = {"name": resolved_gateway_name}
    if oidc is not None:
        spec["oidc"] = oidc
    return spec


def bootstrap_gateway_ref(
    aitenant_name: str,
    aitenant_spec: dict[str, Any],
) -> tuple[str, str]:
    """Resolve the bootstrap Gateway name and namespace for an AITenant spec."""
    gateway_spec = aitenant_spec.get("gateway", {})
    return (
        gateway_spec.get("name", aitenant_name),
        MAAS_GATEWAY_NAMESPACE,
    )


def bootstrap_gateway_ref_from_aitenant(aitenant: AITenant) -> tuple[str, str]:
    """Resolve the bootstrap Gateway name and namespace from AITenant status or spec."""
    if aitenant.exists:
        fresh_aitenant = _fresh_aitenant(aitenant=aitenant)
        status_gateway_ref = getattr(fresh_aitenant.instance.status, "gatewayRef", None)
        if status_gateway_ref is not None:
            return status_gateway_ref.name, status_gateway_ref.namespace
    aitenant_spec: dict[str, Any] = {}
    if aitenant.gateway is not None:
        aitenant_spec["gateway"] = aitenant.gateway
    return bootstrap_gateway_ref(
        aitenant_name=aitenant.name,
        aitenant_spec=aitenant_spec,
    )


def aitenant_from_spec(
    admin_client: DynamicClient,
    aitenant_name: str,
    cr_namespace: str,
    aitenant_spec: dict[str, Any],
    teardown: bool = False,
) -> AITenant:
    """Return an AITenant configured from spec; use with ``with aitenant_from_spec(...) as aitenant:``."""
    aitenant_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": aitenant_name,
        "namespace": cr_namespace,
        "teardown": teardown,
        "wait_for_resource": True,
    }
    if "gateway" in aitenant_spec:
        aitenant_kwargs["gateway"] = aitenant_spec["gateway"]
    if "oidc" in aitenant_spec:
        aitenant_kwargs["oidc"] = aitenant_spec["oidc"]
    return AITenant(**aitenant_kwargs)


def aitenant_bootstrap_gateway(
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
def bootstrap_gateway_context(
    admin_client: DynamicClient,
    gateway_name: str,
    gateway_namespace: str,
    teardown: bool,
) -> Generator[Gateway]:
    """Yield a pre-provisioned bootstrap Gateway for the duration of the context."""
    with aitenant_bootstrap_gateway(
        admin_client=admin_client,
        gateway_name=gateway_name,
        gateway_namespace=gateway_namespace,
        teardown=teardown,
    ) as gateway:
        yield gateway


def deploy_and_verify_aitenant_ready(aitenant: AITenant) -> None:
    """Create the AITenant CR if missing and wait until it reports Ready with phase Active."""
    if not aitenant.exists:
        aitenant.deploy()
    verify_aitenant_ready(aitenant=aitenant)


def build_aitenant_test_context(aitenant: AITenant) -> AITenantTestContext:
    """Build the standard test context dict from a deployed AITenant."""
    return AITenantTestContext(
        aitenant=aitenant,
        aitenant_name=aitenant.name,
        tenant_namespace_name=tenant_namespace_name_from_aitenant(aitenant=aitenant),
    )


def verify_aitenant_ready(aitenant: AITenant) -> None:
    """Assert the AITenant exists and reports Ready=True with phase Active."""
    assert aitenant.exists, f"AITenant '{aitenant.name}' not found in namespace '{aitenant.namespace}'"
    aitenant.wait_for_condition(condition="Ready", status="True", timeout=300)
    phase = getattr(aitenant.instance.status, "phase", "") or ""
    assert phase == "Active", f"Expected AITenant phase Active, got '{phase}'"


def verify_aitenant_bootstrap_children(
    admin_client: DynamicClient,
    test_context: AITenantTestContext,
    infra_namespace: str = AITENANT_INFRA_NAMESPACE,
) -> None:
    """Assert AITenant bootstrap created the expected namespace, Gateway, and MaasTenantConfig.

    AITenant status.gatewayRef tracks the per-tenant pre-provisioned bootstrap gateway.
    The controller also creates MaasTenantConfig/default-tenant in the tenant namespace.
    """
    aitenant = test_context["aitenant"]
    aitenant_name = test_context["aitenant_name"]
    tenant_namespace_name = test_context["tenant_namespace_name"]

    fresh_aitenant = _fresh_aitenant(aitenant=aitenant)
    aitenant_status = fresh_aitenant.instance.status
    status_gateway_ref = getattr(aitenant_status, "gatewayRef", None)
    assert status_gateway_ref is not None, f"AITenant '{aitenant_name}' status.gatewayRef should be set after bootstrap"
    gateway_name = status_gateway_ref.name
    gateway_namespace = status_gateway_ref.namespace
    status_tenant_namespace = getattr(aitenant_status, "tenantNamespace", None)
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
    assert namespace_labels.get(AIGATEWAY_TENANT_LABEL) == aitenant_name, (
        f"Tenant namespace '{tenant_namespace_name}' label {AIGATEWAY_TENANT_LABEL} expected {aitenant_name!r}, "
        f"got {namespace_labels.get(AIGATEWAY_TENANT_LABEL)!r}"
    )
    assert namespace_annotations.get(AIGATEWAY_NAME_ANNOTATION) == aitenant_name, (
        f"Tenant namespace {AIGATEWAY_NAME_ANNOTATION} expected {aitenant_name!r}, "
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

    bootstrapped_tenant_config = MaasTenantConfig(
        client=admin_client,
        name=AIGATEWAY_BOOTSTRAPPED_TENANT_NAME,
        namespace=tenant_namespace_name,
        ensure_exists=True,
    )
    assert bootstrapped_tenant_config.exists, (
        f"MaasTenantConfig/{AIGATEWAY_BOOTSTRAPPED_TENANT_NAME} was not created in '{tenant_namespace_name}'"
    )
    tenant_config_labels = dict(bootstrapped_tenant_config.instance.metadata.labels or {})
    assert tenant_config_labels.get(AIGATEWAY_MANAGED_BY_LABEL) is not None, (
        f"MaasTenantConfig/{AIGATEWAY_BOOTSTRAPPED_TENANT_NAME} should have label {AIGATEWAY_MANAGED_BY_LABEL}"
    )
    verify_maas_tenant_config_ready(maas_tenant_config=bootstrapped_tenant_config)
    LOGGER.info(
        f"AITenant '{aitenant_name}' bootstrap verified: namespace, gateway, and "
        f"MaasTenantConfig/{AIGATEWAY_BOOTSTRAPPED_TENANT_NAME} exist with expected metadata"
    )


def verify_bootstrapped_tenant_oidc(
    admin_client: DynamicClient,
    tenant_namespace_name: str,
    expected_oidc: dict[str, Any],
) -> None:
    """Assert bootstrapped Tenant externalOIDC mirrors the AITenant oidc spec."""
    bootstrapped_tenant = Tenant(
        client=admin_client,
        name=AIGATEWAY_BOOTSTRAPPED_TENANT_NAME,
        namespace=tenant_namespace_name,
        ensure_exists=True,
    )
    tenant_oidc = bootstrapped_tenant.instance.spec.externalOIDC
    assert tenant_oidc is not None, (
        f"Tenant/{AIGATEWAY_BOOTSTRAPPED_TENANT_NAME} in '{tenant_namespace_name}' "
        "should mirror AITenant oidc into externalOIDC"
    )
    for field_name, expected_value in expected_oidc.items():
        actual_value = getattr(tenant_oidc, field_name, None)
        assert actual_value == expected_value, (
            f"Tenant externalOIDC.{field_name} expected {expected_value!r}, got {actual_value!r}"
        )


def _normalize_rbac_subjects(subjects: list[Any]) -> list[dict[str, str]]:
    """Return RoleBinding subjects as kind/name pairs for assertions."""
    return [{"kind": subject.kind, "name": subject.name} for subject in subjects]


def verify_aitenant_role_binding(
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


def tenant_admin_role_binding_name(aitenant_name: str) -> str:
    """Return a test RoleBinding name for tenant-admin access in the tenant namespace."""
    return f"{tenant_admin_role_name(aitenant_name=aitenant_name)}-admins"


def object_admin_role_binding_name(aitenant_name: str) -> str:
    """Return a test RoleBinding name for object-admin access in the infra namespace."""
    return f"{aitenant_object_admin_role_name(aitenant_name=aitenant_name)}-admins"


@contextmanager
def aitenant_admin_role_bindings(
    admin_client: DynamicClient,
    aitenant_name: str,
    tenant_namespace_name: str,
    infra_namespace: str,
    subjects: list[dict[str, str]],
    teardown: bool = True,
) -> Generator[tuple[RoleBinding, RoleBinding], Any, Any]:
    """Create manual tenant-admin and object-admin RoleBindings for the given subjects."""
    tenant_admin_name = tenant_admin_role_name(aitenant_name=aitenant_name)
    object_admin_name = aitenant_object_admin_role_name(aitenant_name=aitenant_name)
    tenant_binding_name = tenant_admin_role_binding_name(aitenant_name=aitenant_name)
    object_binding_name = object_admin_role_binding_name(aitenant_name=aitenant_name)
    if len(subjects) != 1:
        raise ValueError("aitenant_admin_role_bindings currently supports exactly one RBAC subject")
    subject = subjects[0]
    with (
        RoleBinding(
            client=admin_client,
            namespace=tenant_namespace_name,
            name=tenant_binding_name,
            role_ref_name=tenant_admin_name,
            role_ref_kind="Role",
            subjects_kind=subject["kind"],
            subjects_name=subject["name"],
            teardown=teardown,
        ) as tenant_role_binding,
        RoleBinding(
            client=admin_client,
            namespace=infra_namespace,
            name=object_binding_name,
            role_ref_name=object_admin_name,
            role_ref_kind="Role",
            subjects_kind=subject["kind"],
            subjects_name=subject["name"],
            teardown=teardown,
        ) as object_role_binding,
    ):
        yield tenant_role_binding, object_role_binding


def verify_aitenant_controller_creates_admin_roles_only(
    admin_client: DynamicClient,
    aitenant_name: str,
    tenant_namespace_name: str,
    infra_namespace: str,
) -> None:
    """Assert the controller creates admin Roles but does not create RoleBindings."""
    tenant_admin_name = tenant_admin_role_name(aitenant_name=aitenant_name)
    object_admin_name = aitenant_object_admin_role_name(aitenant_name=aitenant_name)
    tenant_role = Role(client=admin_client, name=tenant_admin_name, namespace=tenant_namespace_name)
    infra_role = Role(client=admin_client, name=object_admin_name, namespace=infra_namespace)
    assert tenant_role.exists, f"Role '{tenant_namespace_name}/{tenant_admin_name}' should exist"
    assert infra_role.exists, f"Role '{infra_namespace}/{object_admin_name}' should exist"
    verify_aitenant_role_binding(
        admin_client=admin_client,
        namespace=tenant_namespace_name,
        binding_name=tenant_admin_name,
        role_name=tenant_admin_name,
        should_exist=False,
    )
    verify_aitenant_role_binding(
        admin_client=admin_client,
        namespace=infra_namespace,
        binding_name=object_admin_name,
        role_name=object_admin_name,
        should_exist=False,
    )


def verify_manual_aitenant_admin_role_bindings(
    admin_client: DynamicClient,
    aitenant_name: str,
    tenant_namespace_name: str,
    infra_namespace: str,
    expected_subjects: list[dict[str, str]],
) -> None:
    """Assert manually created tenant-admin and object-admin RoleBindings reference controller Roles."""
    tenant_admin_name = tenant_admin_role_name(aitenant_name=aitenant_name)
    object_admin_name = aitenant_object_admin_role_name(aitenant_name=aitenant_name)
    verify_aitenant_role_binding(
        admin_client=admin_client,
        namespace=tenant_namespace_name,
        binding_name=tenant_admin_role_binding_name(aitenant_name=aitenant_name),
        role_name=tenant_admin_name,
        expected_subjects=expected_subjects,
    )
    verify_aitenant_role_binding(
        admin_client=admin_client,
        namespace=infra_namespace,
        binding_name=object_admin_role_binding_name(aitenant_name=aitenant_name),
        role_name=object_admin_name,
        expected_subjects=expected_subjects,
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


def verify_aitenant_rbac_children_removed(
    admin_client: DynamicClient,
    aitenant_name: str,
    tenant_namespace_name: str,
    infra_namespace: str,
    timeout: int = 300,
) -> None:
    """Assert controller-owned tenant-admin and object-admin Roles were removed after AITenant deletion."""
    tenant_admin_name = tenant_admin_role_name(aitenant_name=aitenant_name)
    object_admin_name = aitenant_object_admin_role_name(aitenant_name=aitenant_name)
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
            Role(
                client=admin_client,
                name=object_admin_name,
                namespace=infra_namespace,
            ).exists
        ),
        resource_label=f"Role '{infra_namespace}/{object_admin_name}'",
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


def verify_aitenant_bootstrap_children_removed(
    admin_client: DynamicClient,
    test_context: AITenantTestContext,
    infra_namespace: str = AITENANT_INFRA_NAMESPACE,
    timeout: int = 300,
) -> None:
    """Assert controller-owned MaasTenantConfig and RBAC children were removed after AITenant deletion."""
    aitenant = test_context["aitenant"]
    aitenant_name = test_context["aitenant_name"]
    tenant_namespace_name = test_context["tenant_namespace_name"]
    gateway_name, gateway_namespace = bootstrap_gateway_ref_from_aitenant(aitenant=aitenant)

    verify_preprovisioned_bootstrap_gateway_preserved(
        admin_client=admin_client,
        gateway_name=gateway_name,
        gateway_namespace=gateway_namespace,
    )

    _wait_until_resource_absent(
        exists_check=lambda: (
            MaasTenantConfig(
                client=admin_client,
                name=AIGATEWAY_BOOTSTRAPPED_TENANT_NAME,
                namespace=tenant_namespace_name,
            ).exists
        ),
        resource_label=(f"MaasTenantConfig/{AIGATEWAY_BOOTSTRAPPED_TENANT_NAME} in '{tenant_namespace_name}'"),
        timeout=timeout,
    )

    verify_aitenant_rbac_children_removed(
        admin_client=admin_client,
        aitenant_name=aitenant_name,
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


def _fresh_aitenant(aitenant: AITenant) -> AITenant:
    """Return a new handle to re-read the current AITenant status from the API."""
    return AITenant(
        client=aitenant.client,
        name=aitenant.name,
        namespace=aitenant.namespace,
        wait_for_resource=False,
    )


def get_aitenant_ready_reason(aitenant: AITenant) -> str:
    """Return the Ready condition reason, or an empty string when absent."""
    fresh_aitenant = _fresh_aitenant(aitenant=aitenant)
    status = getattr(fresh_aitenant.instance, "status", {}) or {}
    for condition in status.get("conditions", []):
        if condition.get("type") == "Ready":
            return condition.get("reason") or ""
    return ""


def aitenant_has_status(
    aitenant: AITenant,
    phase: str,
    ready_reason: str | None = None,
) -> bool:
    """Return True when AITenant status matches the expected phase and optional Ready reason."""
    fresh_aitenant = _fresh_aitenant(aitenant=aitenant)
    current_phase = getattr(fresh_aitenant.instance.status, "phase", "") or ""
    if current_phase != phase:
        return False
    if ready_reason is None:
        return True
    return get_aitenant_ready_reason(aitenant=aitenant) == ready_reason


def wait_until_aitenant_status(
    aitenant: AITenant,
    phase: str,
    ready_reason: str | None = None,
    timeout: int = 120,
) -> None:
    """Wait until AITenant reaches the expected phase and optional Ready reason."""
    try:
        for matched in TimeoutSampler(
            wait_timeout=timeout,
            sleep=5,
            func=lambda: aitenant_has_status(
                aitenant=aitenant,
                phase=phase,
                ready_reason=ready_reason,
            ),
        ):
            if matched:
                return
    except TimeoutExpiredError:
        current_phase = getattr(_fresh_aitenant(aitenant=aitenant).instance.status, "phase", "") or ""
        current_reason = get_aitenant_ready_reason(aitenant=aitenant)
        pytest.fail(
            f"AITenant '{aitenant.name}' did not reach phase={phase} "
            f"ready_reason={ready_reason}: phase={current_phase} ready_reason={current_reason}"
        )


def verify_derived_tenant_namespace_name(
    aitenant: AITenant,
    expected_tenant_namespace_name: str,
) -> None:
    """Assert status.tenantNamespace matches the controller-derived tenant namespace."""
    actual_tenant_namespace_name = tenant_namespace_name_from_aitenant(aitenant=aitenant)
    assert actual_tenant_namespace_name == expected_tenant_namespace_name, (
        f"AITenant status.tenantNamespace expected {expected_tenant_namespace_name!r}, "
        f"got {actual_tenant_namespace_name!r}"
    )


def verify_default_maas_tenant_unaffected(admin_client: DynamicClient) -> None:
    """Assert the cluster default-tenant MaasTenantConfig in models-as-a-service is still Ready."""
    default_maas_tenant_config = MaasTenantConfig(
        client=admin_client,
        name=AIGATEWAY_BOOTSTRAPPED_TENANT_NAME,
        namespace=MAAS_SUBSCRIPTION_NAMESPACE,
    )
    verify_maas_tenant_config_ready(maas_tenant_config=default_maas_tenant_config)
    LOGGER.info(
        f"Regression check passed: MaasTenantConfig/{AIGATEWAY_BOOTSTRAPPED_TENANT_NAME} in "
        f"'{MAAS_SUBSCRIPTION_NAMESPACE}' is still Ready"
    )
