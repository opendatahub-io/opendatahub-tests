import hashlib
import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, TypedDict

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.deployment import Deployment
from ocp_resources.gateway_gateway_networking_k8s_io import Gateway
from ocp_resources.namespace import Namespace
from ocp_resources.resource import NamespacedResource, ResourceEditor
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_gateway.models_as_a_service.maas_subscription.utils import MAAS_SUBSCRIPTION_NAMESPACE
from tests.ai_gateway.models_as_a_service.observability.utils import (
    MAAS_CONTROLLER_DEPLOYMENT_NAME,
    get_maas_controller_env_var,
)
from tests.ai_gateway.models_as_a_service.utils import (
    AIGATEWAY_GATEWAY_CLASS_NAME,
    AITENANT_INFRA_NAMESPACE,
    bootstrap_gateway_ref_from_aitenant,
    fresh_aitenant,
    verify_maas_gateway_programmed,
    verify_maas_tenant_config_ready,
)
from utilities.constants import MAAS_GATEWAY_NAME, ApiGroups
from utilities.general import generate_random_name
from utilities.resources.aitenant import AITenant
from utilities.resources.maastenantconfig import MaasTenantConfig

LOGGER = structlog.get_logger(name=__name__)

AITENANT_CRD_NAME = f"aitenants.{ApiGroups.MAAS_IO}"
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
AIGATEWAY_MANAGED_BY_LABEL = "maas.opendatahub.io/managed-by-aitenant"
AIGATEWAY_TENANT_LABEL = "ai-gateway.opendatahub.io/tenant"
GATEWAY_ACCESS_LABEL = "maas.opendatahub.io/gateway-access"
GATEWAY_ACCESS_LABEL_VALUE = "true"

ENABLE_TENANT_NAMESPACE_DISCOVERY_ENV = "ENABLE_TENANT_NAMESPACE_DISCOVERY"
DISCOVERY_CONTROLLER_ARG = "--enable-tenant-namespace-discovery=true"
DISCOVERY_CONTROLLER_ARG_PREFIX = "--enable-tenant-namespace-discovery"

LABEL_TENANT_NAME = "maas.opendatahub.io/tenant-name"
LABEL_TENANT_NAMESPACE = "maas.opendatahub.io/tenant-namespace"

FINALIZER_AUTH_POLICY = "maas.opendatahub.io/authpolicy-cleanup"
FINALIZER_SUBSCRIPTION = "maas.opendatahub.io/subscription-cleanup"
FINALIZER_MODEL_REF = "maas.opendatahub.io/model-cleanup"

MODEL_REF_RECONCILED_PHASES = ("Pending", "Active", "Degraded")

SUBSCRIPTION_RECONCILED_PHASES = ("Active", "Degraded")


class AITenantTestContext(TypedDict):
    aitenant: AITenant
    aitenant_name: str
    tenant_namespace_name: str


class AITenantPreexistingNamespaceContext(TypedDict):
    aitenant: AITenant
    tenant_namespace: Namespace
    tenant_namespace_name: str


class TenantNamespaceDiscoveryCase(TypedDict):
    suffix: str
    tenant_namespace_name: str
    tenant_label_name: str
    policy_name: str
    subscription_name: str
    model_ref_name: str


def expected_tenant_namespace_name(aitenant_name: str) -> str:
    """Return the tenant namespace name the controller derives for an AITenant."""
    if aitenant_name == MAAS_SUBSCRIPTION_NAMESPACE:
        return MAAS_SUBSCRIPTION_NAMESPACE
    return f"{AITENANT_TENANT_NAMESPACE_PREFIX}{aitenant_name}"


def tenant_namespace_name_from_aitenant(aitenant: AITenant) -> str:
    """Return the reconciled tenant namespace name from AITenant status."""
    refreshed_aitenant = fresh_aitenant(aitenant=aitenant)
    status_tenant_namespace = getattr(refreshed_aitenant.instance.status, "tenantNamespace", None)
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


def build_aitenant_test_context(aitenant: AITenant) -> AITenantTestContext:
    """Build the standard test context dict from a deployed AITenant."""
    return AITenantTestContext(
        aitenant=aitenant,
        aitenant_name=aitenant.name,
        tenant_namespace_name=tenant_namespace_name_from_aitenant(aitenant=aitenant),
    )


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

    refreshed_aitenant = fresh_aitenant(aitenant=aitenant)
    aitenant_status = refreshed_aitenant.instance.status
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
    verify_tenant_namespace_gateway_access_label_present(
        admin_client=admin_client,
        tenant_namespace_name=tenant_namespace_name,
        namespace_labels=namespace_labels,
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


def verify_aitenant_oidc_stays_in_spec(
    aitenant: AITenant,
    expected_oidc: dict[str, Any],
) -> None:
    """Assert AITenant.spec.oidc remains on the AITenant (not copied to Tenant/MaasTenantConfig)."""
    refreshed_aitenant = fresh_aitenant(aitenant=aitenant)
    aitenant_oidc = getattr(refreshed_aitenant.instance.spec, "oidc", None)
    assert aitenant_oidc is not None, f"AITenant '{aitenant.namespace}/{aitenant.name}' should retain spec.oidc"
    for field_name, expected_value in expected_oidc.items():
        actual_value = getattr(aitenant_oidc, field_name, None)
        assert actual_value == expected_value, (
            f"AITenant spec.oidc.{field_name} expected {expected_value!r}, got {actual_value!r}"
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


def delete_aitenant_and_wait(aitenant: AITenant, timeout: int = 300) -> None:
    """Delete an AITenant CR and wait until it is removed from the API."""
    aitenant.delete()
    aitenant.wait_deleted(timeout=timeout)


def verify_gateway_access_label_removed_after_aitenant_delete(
    admin_client: DynamicClient,
    tenant_namespace_name: str,
    aitenant: AITenant,
    timeout: int = 300,
) -> None:
    """Assert gateway-access is present before delete and absent after AITenant deletion."""
    verify_tenant_namespace_gateway_access_label_present(
        admin_client=admin_client,
        tenant_namespace_name=tenant_namespace_name,
    )
    delete_aitenant_and_wait(aitenant=aitenant, timeout=timeout)
    verify_tenant_namespace_preserved(
        admin_client=admin_client,
        tenant_namespace_name=tenant_namespace_name,
    )
    verify_tenant_namespace_gateway_access_label_absent(
        admin_client=admin_client,
        tenant_namespace_name=tenant_namespace_name,
    )


def verify_tenant_namespace_discovery_labels_present(
    admin_client: DynamicClient,
    tenant_namespace_name: str,
) -> None:
    """Assert the tenant namespace carries maas-controller tenant namespace discovery labels."""
    tenant_namespace = Namespace(
        client=admin_client,
        name=tenant_namespace_name,
        ensure_exists=True,
    )
    namespace_labels = dict(tenant_namespace.instance.metadata.labels or {})
    assert namespace_labels.get(AIGATEWAY_MANAGED_BY_LABEL) == "true", (
        f"Tenant namespace '{tenant_namespace_name}' missing label {AIGATEWAY_MANAGED_BY_LABEL}='true'"
    )
    assert namespace_labels.get(LABEL_TENANT_NAMESPACE) == tenant_namespace_name, (
        f"Tenant namespace '{tenant_namespace_name}' label {LABEL_TENANT_NAMESPACE} expected "
        f"{tenant_namespace_name!r}, got {namespace_labels.get(LABEL_TENANT_NAMESPACE)!r}"
    )
    tenant_label = namespace_labels.get(AIGATEWAY_TENANT_LABEL)
    assert tenant_label, f"Tenant namespace '{tenant_namespace_name}' missing label {AIGATEWAY_TENANT_LABEL}"
    assert namespace_labels.get(LABEL_TENANT_NAME) == tenant_label, (
        f"Tenant namespace '{tenant_namespace_name}' label {LABEL_TENANT_NAME} expected "
        f"{tenant_label!r}, got {namespace_labels.get(LABEL_TENANT_NAME)!r}"
    )


def verify_tenant_namespace_discovery_labels_absent(
    admin_client: DynamicClient,
    tenant_namespace_name: str,
) -> None:
    """Assert tenant namespace discovery labels were removed from the namespace."""
    tenant_namespace = Namespace(
        client=admin_client,
        name=tenant_namespace_name,
        ensure_exists=True,
    )
    namespace_labels = dict(tenant_namespace.instance.metadata.labels or {})
    assert namespace_labels.get(AIGATEWAY_MANAGED_BY_LABEL) is None, (
        f"Tenant namespace '{tenant_namespace_name}' should not retain {AIGATEWAY_MANAGED_BY_LABEL}"
    )
    assert namespace_labels.get(AIGATEWAY_TENANT_LABEL) is None, (
        f"Tenant namespace '{tenant_namespace_name}' should not retain {AIGATEWAY_TENANT_LABEL}"
    )
    assert namespace_labels.get(LABEL_TENANT_NAME) is None, (
        f"Tenant namespace '{tenant_namespace_name}' should not retain {LABEL_TENANT_NAME}"
    )
    assert namespace_labels.get(LABEL_TENANT_NAMESPACE) is None, (
        f"Tenant namespace '{tenant_namespace_name}' should not retain {LABEL_TENANT_NAMESPACE}"
    )


def verify_tenant_namespace_gateway_access_label_present(
    admin_client: DynamicClient,
    tenant_namespace_name: str,
    namespace_labels: dict[str, str] | None = None,
) -> None:
    """Assert the tenant namespace has maas.opendatahub.io/gateway-access=true."""
    if namespace_labels is None:
        tenant_namespace = Namespace(
            client=admin_client,
            name=tenant_namespace_name,
            ensure_exists=True,
        )
        namespace_labels = dict(tenant_namespace.instance.metadata.labels or {})
    assert namespace_labels.get(GATEWAY_ACCESS_LABEL) == GATEWAY_ACCESS_LABEL_VALUE, (
        f"Tenant namespace '{tenant_namespace_name}' label {GATEWAY_ACCESS_LABEL} expected "
        f"{GATEWAY_ACCESS_LABEL_VALUE!r}, got {namespace_labels.get(GATEWAY_ACCESS_LABEL)!r}"
    )


def verify_tenant_namespace_gateway_access_label_absent(
    admin_client: DynamicClient,
    tenant_namespace_name: str,
) -> None:
    """Assert maas.opendatahub.io/gateway-access was removed from the tenant namespace."""
    tenant_namespace = Namespace(
        client=admin_client,
        name=tenant_namespace_name,
        ensure_exists=True,
    )
    labels = tenant_namespace.instance.metadata.labels or {}
    assert labels.get(GATEWAY_ACCESS_LABEL) is None, (
        f"Tenant namespace '{tenant_namespace_name}' should not retain {GATEWAY_ACCESS_LABEL}"
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
    assert labels.get(GATEWAY_ACCESS_LABEL) is None, (
        f"Tenant namespace '{tenant_namespace_name}' should not retain {GATEWAY_ACCESS_LABEL}"
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


def get_aitenant_ready_reason(aitenant: AITenant) -> str:
    """Return the Ready condition reason, or an empty string when absent."""
    refreshed_aitenant = fresh_aitenant(aitenant=aitenant)
    status = getattr(refreshed_aitenant.instance, "status", {}) or {}
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
    refreshed_aitenant = fresh_aitenant(aitenant=aitenant)
    current_phase = getattr(refreshed_aitenant.instance.status, "phase", "") or ""
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
        current_phase = getattr(fresh_aitenant(aitenant=aitenant).instance.status, "phase", "") or ""
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


def build_tenant_namespace_discovery_case() -> TenantNamespaceDiscoveryCase:
    """Return unique resource names for a tenant namespace discovery test case."""
    suffix = generate_random_name()[:8]
    tenant_label_name = f"e2e-mt-{suffix}"
    tenant_namespace_name = f"ai-tenant-{tenant_label_name}"
    return TenantNamespaceDiscoveryCase(
        suffix=suffix,
        tenant_namespace_name=tenant_namespace_name,
        tenant_label_name=tenant_label_name,
        policy_name=f"e2e-policy-{suffix}",
        subscription_name=f"e2e-sub-{suffix}",
        model_ref_name=f"e2e-model-ref-{suffix}",
    )


def refresh_maas_namespaced_resource(resource: NamespacedResource) -> NamespacedResource:
    """Return a new handle with an up-to-date instance from the API."""
    return type(resource)(
        client=resource.client,
        name=resource.name,
        namespace=resource.namespace,
        wait_for_resource=False,
    )


def maas_model_ref_runtime_ready(model_ref: NamespacedResource) -> bool:
    """Return True when MaaSModelRef status reports RuntimeReady=True."""
    refreshed_model_ref = refresh_maas_namespaced_resource(resource=model_ref)
    status = refreshed_model_ref.instance.status
    if status is None:
        return False
    conditions = getattr(status, "conditions", None)
    if not conditions:
        return False
    for condition in conditions:
        condition_type = getattr(condition, "type", None)
        condition_status = getattr(condition, "status", None)
        if condition_type == "RuntimeReady" and condition_status == "True":
            return True
    return False


def _tenant_namespace_discovery_from_controller_arg(arg: str) -> bool | None:
    """Return discovery on/off when arg sets the flag, else None if the arg is unrelated."""
    if arg == DISCOVERY_CONTROLLER_ARG_PREFIX:
        return True
    if arg.startswith(f"{DISCOVERY_CONTROLLER_ARG_PREFIX}="):
        value = arg.split("=", maxsplit=1)[1].lower()
        return value in {"1", "true", "yes", "on"}
    return None


def maas_controller_tenant_namespace_discovery_enabled(admin_client: DynamicClient) -> bool:
    """Return True when maas-controller is started with tenant namespace discovery enabled."""
    env_value = get_maas_controller_env_var(
        admin_client=admin_client,
        env_name=ENABLE_TENANT_NAMESPACE_DISCOVERY_ENV,
    )
    if env_value.lower() in {"1", "true", "yes", "on"}:
        return True

    applications_namespace = py_config["applications_namespace"]
    controller_deployment = Deployment(
        client=admin_client,
        name=MAAS_CONTROLLER_DEPLOYMENT_NAME,
        namespace=applications_namespace,
        ensure_exists=True,
    )
    discovery_from_args: bool | None = None
    for container in controller_deployment.instance.spec.template.spec.containers:
        for arg in container.args or []:
            parsed = _tenant_namespace_discovery_from_controller_arg(arg=arg)
            if parsed is not None:
                discovery_from_args = parsed
    return discovery_from_args is True


def require_tenant_namespace_discovery_enabled(admin_client: DynamicClient) -> None:
    """Skip or fail when tenant namespace discovery is not enabled on maas-controller."""
    discovery_enabled = maas_controller_tenant_namespace_discovery_enabled(admin_client=admin_client)
    env_requires_discovery = os.environ.get(ENABLE_TENANT_NAMESPACE_DISCOVERY_ENV, "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if env_requires_discovery and not discovery_enabled:
        pytest.fail(
            f"{ENABLE_TENANT_NAMESPACE_DISCOVERY_ENV}=true but maas-controller is missing "
            f"{DISCOVERY_CONTROLLER_ARG}; patch the deployment to run tenant namespace discovery tests"
        )
    if not discovery_enabled:
        pytest.skip(
            f"maas-controller does not have {DISCOVERY_CONTROLLER_ARG}; "
            f"set {ENABLE_TENANT_NAMESPACE_DISCOVERY_ENV}=true and patch the deployment to run these tests"
        )


def require_aitenant_crd_for_discovery(admin_client: DynamicClient) -> None:
    """Skip or fail when the AITenant CRD is not installed (tenant namespace discovery tests)."""
    aitenant_crd = CustomResourceDefinition(client=admin_client, name=AITENANT_CRD_NAME)
    if aitenant_crd.exists:
        return
    env_requires_discovery = os.environ.get(ENABLE_TENANT_NAMESPACE_DISCOVERY_ENV, "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if env_requires_discovery:
        pytest.fail(f"Missing CRD {AITENANT_CRD_NAME}; tenant namespace discovery tests cannot run")
    pytest.skip(f"Missing CRD {AITENANT_CRD_NAME}; AITenant is not applicable on this cluster")


def discovery_namespace_label_patch(
    tenant_label_name: str,
    tenant_namespace_name: str,
) -> dict[str, str]:
    """Return namespace labels that mark a namespace for tenant discovery reconciliation."""
    return {
        AIGATEWAY_TENANT_LABEL: tenant_label_name,
        AIGATEWAY_MANAGED_BY_LABEL: "true",
        LABEL_TENANT_NAME: tenant_label_name,
        LABEL_TENANT_NAMESPACE: tenant_namespace_name,
    }


def discovery_namespace_label_removal_patch() -> dict[str, str | None]:
    """Return a merge patch that removes tenant discovery labels from a namespace."""
    return {
        AIGATEWAY_TENANT_LABEL: None,
        AIGATEWAY_MANAGED_BY_LABEL: None,
        LABEL_TENANT_NAME: None,
        LABEL_TENANT_NAMESPACE: None,
    }


def apply_discovery_namespace_labels(
    admin_client: DynamicClient,
    tenant_namespace_name: str,
    tenant_label_name: str,
) -> None:
    """Apply discovery labels to an existing tenant namespace."""
    namespace = Namespace(client=admin_client, name=tenant_namespace_name, ensure_exists=True)
    ResourceEditor(
        patches={
            namespace: {
                "metadata": {
                    "labels": discovery_namespace_label_patch(
                        tenant_label_name=tenant_label_name,
                        tenant_namespace_name=tenant_namespace_name,
                    ),
                },
            },
        },
    ).update()
    LOGGER.info(f"Applied tenant discovery labels to namespace '{tenant_namespace_name}'")


def remove_discovery_namespace_labels(admin_client: DynamicClient, tenant_namespace_name: str) -> None:
    """Remove discovery labels from a tenant namespace."""
    namespace = Namespace(client=admin_client, name=tenant_namespace_name, ensure_exists=True)
    ResourceEditor(
        patches={namespace: {"metadata": {"labels": discovery_namespace_label_removal_patch()}}},
    ).update()
    LOGGER.info(f"Removed tenant discovery labels from namespace '{tenant_namespace_name}'")


def prepare_discovered_tenant_namespace(
    admin_client: DynamicClient,
    tenant_namespace_name: str,
    tenant_label_name: str,
    gateway_name: str = MAAS_GATEWAY_NAME,
) -> None:
    """Label a namespace for discovery and default-gateway HTTPRoute attachment."""
    from tests.ai_gateway.models_as_a_service.multitenancy.utils import label_namespace_gateway_access

    apply_discovery_namespace_labels(
        admin_client=admin_client,
        tenant_namespace_name=tenant_namespace_name,
        tenant_label_name=tenant_label_name,
    )
    label_namespace_gateway_access(
        admin_client=admin_client,
        namespace_name=tenant_namespace_name,
        gateway_name=gateway_name,
    )


@contextmanager
def tenant_namespace_for_discovery(
    admin_client: DynamicClient,
    tenant_namespace_name: str,
    teardown: bool,
) -> Generator[str, Any, Any]:
    """Create an empty tenant namespace for discovery tests."""
    with Namespace(
        client=admin_client,
        name=tenant_namespace_name,
        teardown=teardown,
    ) as tenant_namespace:
        yield tenant_namespace.name


@contextmanager
def default_tenant_maastenantconfig(
    admin_client: DynamicClient,
    tenant_namespace_name: str,
    teardown: bool,
) -> Generator[MaasTenantConfig, Any, Any]:
    """Create MaasTenantConfig/default-tenant so MaaS CRs are admitted in the namespace."""
    with MaasTenantConfig(
        client=admin_client,
        name=AIGATEWAY_BOOTSTRAPPED_TENANT_NAME,
        namespace=tenant_namespace_name,
        teardown=teardown,
        wait_for_resource=True,
    ) as tenant_config:
        yield tenant_config


@contextmanager
def synthetic_discovery_tenant_namespace(
    admin_client: DynamicClient,
    teardown: bool,
    discovery_case: TenantNamespaceDiscoveryCase | None = None,
    discovery_labels_applied: bool = True,
) -> Generator[TenantNamespaceDiscoveryCase, Any, Any]:
    """Create namespace + MaasTenantConfig for discovery tests; optionally apply discovery labels."""
    case = discovery_case if discovery_case is not None else build_tenant_namespace_discovery_case()
    tenant_namespace_name = case["tenant_namespace_name"]
    with (
        tenant_namespace_for_discovery(
            admin_client=admin_client,
            tenant_namespace_name=tenant_namespace_name,
            teardown=teardown,
        ),
        default_tenant_maastenantconfig(
            admin_client=admin_client,
            tenant_namespace_name=tenant_namespace_name,
            teardown=teardown,
        ),
    ):
        if discovery_labels_applied:
            prepare_discovered_tenant_namespace(
                admin_client=admin_client,
                tenant_namespace_name=tenant_namespace_name,
                tenant_label_name=case["tenant_label_name"],
            )
        yield case


def wait_for_maas_model_ref_discovered(
    model_ref: NamespacedResource,
    timeout: int = 180,
) -> None:
    """Wait until MaaSModelRef is adopted by maas-controller in a discovery-labeled namespace.

    Tenant-local refs may stay Pending without an auth policy + subscription pairing;
    discovery reconciliation is indicated by the controller finalizer and status.phase.
    """
    wait_for_maas_resource_finalizer(
        resource=model_ref,
        expected_finalizer=FINALIZER_MODEL_REF,
        timeout=timeout,
    )
    wait_for_maas_resource_phase(
        resource=model_ref,
        expected_phases=MODEL_REF_RECONCILED_PHASES,
        timeout=timeout,
    )


def wait_for_maas_auth_policy_active(
    auth_policy: NamespacedResource,
    timeout: int = 180,
) -> None:
    """Wait until MaaSAuthPolicy has the controller finalizer and status.phase Active."""
    wait_for_maas_resource_finalizer(
        resource=auth_policy,
        expected_finalizer=FINALIZER_AUTH_POLICY,
        timeout=timeout,
    )
    phase = wait_for_maas_resource_phase(
        resource=auth_policy,
        expected_phases=("Active",),
        timeout=timeout,
    )
    assert phase == "Active"


def _maas_resource_reconciled_by_controller(
    resource: NamespacedResource,
    controller_finalizer: str,
) -> bool:
    """Return True when maas-controller has started reconciling the resource."""
    finalizers = read_maas_resource_finalizers(resource=resource)
    phase = read_maas_resource_status_phase(resource=resource)
    return controller_finalizer in finalizers or phase is not None


def wait_until_maas_controller_stops_reconciling_discovery_namespace(
    admin_client: DynamicClient,
    tenant_namespace_name: str,
    model_name: str,
    model_namespace: str,
    forbidden_finalizer: str,
    teardown: bool,
    timeout: int = 120,
    probe_unreconciled_seconds: int = 20,
) -> None:
    """Wait until discovery labels are gone and new MaaSAuthPolicies stay unreconciled.

    After discovery labels are removed, the controller informer may briefly still treat the
    namespace as discovered. A short-lived probe policy must stay unreconciled before the
    test creates the subject under assertion.
    """
    from utilities.resources.maa_s_auth_policy import MaaSAuthPolicy

    def namespace_ready_for_negative_assertion() -> bool:
        try:
            verify_tenant_namespace_discovery_labels_absent(
                admin_client=admin_client,
                tenant_namespace_name=tenant_namespace_name,
            )
        except AssertionError:
            return False

        probe_name = f"e2e-discovery-sync-{generate_random_name()[:8]}"
        with MaaSAuthPolicy(
            client=admin_client,
            name=probe_name,
            namespace=tenant_namespace_name,
            model_refs=[{"name": model_name, "namespace": model_namespace}],
            subjects={"groups": [{"name": "system:authenticated"}]},
            teardown=teardown,
            wait_for_resource=True,
        ) as probe_policy:
            try:
                for reconciled in TimeoutSampler(
                    wait_timeout=probe_unreconciled_seconds,
                    sleep=3,
                    func=lambda: _maas_resource_reconciled_by_controller(
                        resource=probe_policy,
                        controller_finalizer=forbidden_finalizer,
                    ),
                ):
                    if reconciled:
                        LOGGER.info(
                            f"Probe MaaSAuthPolicy '{tenant_namespace_name}/{probe_name}' was reconciled; "
                            "waiting for maas-controller informer to catch up"
                        )
                        return False
            except TimeoutExpiredError:
                LOGGER.info(
                    f"Probe MaaSAuthPolicy '{tenant_namespace_name}/{probe_name}' stayed unreconciled for "
                    f"{probe_unreconciled_seconds}s"
                )
                return True
        return False

    try:
        for ready in TimeoutSampler(wait_timeout=timeout, sleep=5, func=namespace_ready_for_negative_assertion):
            if ready:
                LOGGER.info(f"maas-controller stopped reconciling new policies in namespace '{tenant_namespace_name}'")
                return
    except TimeoutExpiredError:
        pytest.fail(
            f"maas-controller still reconciled probe MaaSAuthPolicy in '{tenant_namespace_name}' after "
            f"discovery label removal (timeout {timeout}s); informer may not have caught up"
        )


def read_maas_resource_finalizers(resource: NamespacedResource) -> list[str]:
    """Return the current metadata.finalizers list for a MaaS namespaced resource."""
    refreshed_resource = refresh_maas_namespaced_resource(resource=resource)
    metadata_finalizers = refreshed_resource.instance.metadata.finalizers
    if metadata_finalizers is None:
        return []
    return list(metadata_finalizers)


def read_maas_resource_status_phase(resource: NamespacedResource) -> str | None:
    """Return status.phase when set on a MaaS namespaced resource."""
    refreshed_resource = refresh_maas_namespaced_resource(resource=resource)
    status = refreshed_resource.instance.status
    if status is None:
        return None
    phase = getattr(status, "phase", None)
    if phase is None:
        return None
    return str(phase)


def wait_for_maas_resource_finalizer(
    resource: NamespacedResource,
    expected_finalizer: str,
    timeout: int = 180,
) -> None:
    """Wait until expected_finalizer is present on the resource."""
    resource_label = f"{resource.kind}/{resource.namespace}/{resource.name}"

    def finalizer_present() -> bool:
        return expected_finalizer in read_maas_resource_finalizers(resource=resource)

    try:
        for ready in TimeoutSampler(wait_timeout=timeout, sleep=5, func=finalizer_present):
            if ready:
                LOGGER.info(f"{resource_label} has finalizer {expected_finalizer!r}")
                return
    except TimeoutExpiredError:
        finalizers = read_maas_resource_finalizers(resource=resource)
        pytest.fail(
            f"{resource_label} missing finalizer {expected_finalizer!r} after {timeout}s; finalizers={finalizers}"
        )


def wait_for_maas_resource_phase(
    resource: NamespacedResource,
    expected_phases: tuple[str, ...],
    timeout: int = 180,
) -> str:
    """Wait until status.phase is one of expected_phases."""
    resource_label = f"{resource.kind}/{resource.namespace}/{resource.name}"

    def phase_matches() -> bool:
        phase = read_maas_resource_status_phase(resource=resource)
        return phase is not None and phase in expected_phases

    try:
        for ready in TimeoutSampler(wait_timeout=timeout, sleep=5, func=phase_matches):
            if ready:
                phase = read_maas_resource_status_phase(resource=resource)
                assert phase is not None
                LOGGER.info(f"{resource_label} reached phase {phase!r}")
                return phase
    except TimeoutExpiredError:
        phase = read_maas_resource_status_phase(resource=resource)
        pytest.fail(f"{resource_label} phase not in {expected_phases!r} after {timeout}s; last phase={phase!r}")


def assert_maas_resource_stays_unreconciled(
    resource: NamespacedResource,
    forbidden_finalizer: str,
    timeout: int = 60,
    *,
    treat_runtime_ready_as_reconciled: bool = False,
) -> None:
    """Poll until timeout; fail if maas-controller reconciles (finalizer, phase, or RuntimeReady)."""
    resource_label = f"{resource.kind}/{resource.namespace}/{resource.name}"

    def reconciliation_detected() -> bool:
        finalizers = read_maas_resource_finalizers(resource=resource)
        phase = read_maas_resource_status_phase(resource=resource)
        return (
            forbidden_finalizer in finalizers
            or phase is not None
            or (treat_runtime_ready_as_reconciled and maas_model_ref_runtime_ready(model_ref=resource))
        )

    try:
        for reconciled in TimeoutSampler(
            wait_timeout=timeout,
            sleep=3,
            func=reconciliation_detected,
        ):
            if reconciled:
                finalizers = read_maas_resource_finalizers(resource=resource)
                phase = read_maas_resource_status_phase(resource=resource)
                runtime_ready = treat_runtime_ready_as_reconciled and maas_model_ref_runtime_ready(
                    model_ref=resource,
                )
                pytest.fail(
                    f"{resource_label} was reconciled by maas-controller while discovery labels were absent; "
                    f"finalizers={finalizers}, phase={phase!r}, runtime_ready={runtime_ready}, "
                    f"expected no {forbidden_finalizer!r}, status.phase, or RuntimeReady"
                )
    except TimeoutExpiredError:
        LOGGER.info(f"{resource_label} stayed unreconciled for {timeout}s")
