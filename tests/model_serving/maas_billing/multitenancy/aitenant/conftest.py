from collections.abc import Generator
from typing import Any, TypedDict

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.gateway_gateway_networking_k8s_io import Gateway
from ocp_resources.namespace import Namespace
from pytest_testconfig import config as py_config

from tests.model_serving.maas_billing.maas_subscription.utils import MAAS_SUBSCRIPTION_NAMESPACE
from tests.model_serving.maas_billing.multitenancy.aitenant.utils import (
    AIGATEWAY_NAME_ANNOTATION,
    AIGATEWAY_NAMESPACE_ANNOTATION,
    AITENANT_TEST_OIDC_SPEC,
    AITENANT_TEST_RBAC_ADMINS,
    AITenantPreexistingNamespaceContext,
    AITenantTestContext,
    aitenant_admin_role_bindings,
    aitenant_from_spec,
    bootstrap_gateway_context,
    bootstrap_gateway_ref,
    build_aitenant_spec,
    build_aitenant_test_context,
    deploy_and_verify_aitenant_ready,
    expected_tenant_namespace_name,
)
from utilities.general import generate_random_name
from utilities.resources.aitenant import AITenant


class AITenantTestParams(TypedDict):
    aitenant_name: str
    aitenant_spec: dict[str, Any]


@pytest.fixture
def aitenant_oidc_test_params() -> AITenantTestParams:
    """Return an AITenant name and spec with oidc configured."""
    aitenant_name = f"e2e-aigw-oidc-{generate_random_name()}"
    return AITenantTestParams(
        aitenant_name=aitenant_name,
        aitenant_spec=build_aitenant_spec(aitenant_name=aitenant_name, oidc=AITENANT_TEST_OIDC_SPEC),
    )


@pytest.fixture
def aitenant_oidc_bootstrap_gateway(
    admin_client: DynamicClient,
    aitenant_oidc_test_params: AITenantTestParams,
    teardown_resources: bool,
) -> Generator[Gateway, Any, Any]:
    """Pre-provision the bootstrap Gateway for an oidc AITenant."""
    gateway_name, gateway_namespace = bootstrap_gateway_ref(
        aitenant_name=aitenant_oidc_test_params["aitenant_name"],
        aitenant_spec=aitenant_oidc_test_params["aitenant_spec"],
    )
    with bootstrap_gateway_context(
        admin_client=admin_client,
        gateway_name=gateway_name,
        gateway_namespace=gateway_namespace,
        teardown=teardown_resources,
    ) as gateway:
        yield gateway


@pytest.fixture
def aitenant_oidc(
    admin_client: DynamicClient,
    aitenant_infra_namespace: str,
    aitenant_oidc_test_params: AITenantTestParams,
    aitenant_oidc_bootstrap_gateway: Gateway,
    teardown_resources: bool,
) -> Generator[AITenant, Any, Any]:
    """Deploy an oidc AITenant CR after its bootstrap Gateway exists."""
    with aitenant_from_spec(
        admin_client=admin_client,
        aitenant_name=aitenant_oidc_test_params["aitenant_name"],
        cr_namespace=aitenant_infra_namespace,
        aitenant_spec=aitenant_oidc_test_params["aitenant_spec"],
        teardown=teardown_resources,
    ) as aitenant:
        yield aitenant


@pytest.fixture
def ready_aitenant_oidc(aitenant_oidc: AITenant) -> Generator[AITenant, Any, Any]:
    """Wait until the oidc AITenant reports Ready with phase Active."""
    deploy_and_verify_aitenant_ready(aitenant=aitenant_oidc)
    yield aitenant_oidc


@pytest.fixture
def aitenant_with_oidc(ready_aitenant_oidc: AITenant) -> AITenantTestContext:
    """Return bootstrap test context for a Ready oidc AITenant."""
    return build_aitenant_test_context(aitenant=ready_aitenant_oidc)


@pytest.fixture
def aitenant_with_manual_admin_role_bindings(
    admin_client: DynamicClient,
    aitenant_infra_namespace: str,
    aitenant_for_test: AITenantTestContext,
    teardown_resources: bool,
) -> Generator[AITenantTestContext, Any, Any]:
    """Return a Ready AITenant context with manually created admin RoleBindings."""
    test_context = aitenant_for_test
    with aitenant_admin_role_bindings(
        admin_client=admin_client,
        aitenant_name=test_context["aitenant_name"],
        tenant_namespace_name=test_context["tenant_namespace_name"],
        infra_namespace=aitenant_infra_namespace,
        subjects=AITENANT_TEST_RBAC_ADMINS,
        teardown=teardown_resources,
    ):
        yield test_context


@pytest.fixture
def ready_aitenant_for_deletion(ready_aitenant: AITenant) -> AITenantTestContext:
    """Return bootstrap test context for deletion tests; test owns AITenant deletion."""
    return build_aitenant_test_context(aitenant=ready_aitenant)


@pytest.fixture
def aitenant_preexist_test_params() -> AITenantTestParams:
    """Return an AITenant name and spec for pre-existing tenant namespace adoption."""
    aitenant_name = f"e2e-aigw-preexist-{generate_random_name()}"
    return AITenantTestParams(
        aitenant_name=aitenant_name,
        aitenant_spec=build_aitenant_spec(aitenant_name=aitenant_name),
    )


@pytest.fixture
def aitenant_preexist_bootstrap_gateway(
    admin_client: DynamicClient,
    aitenant_preexist_test_params: AITenantTestParams,
    teardown_resources: bool,
) -> Generator[Gateway, Any, Any]:
    """Pre-provision the bootstrap Gateway for pre-existing namespace adoption."""
    gateway_name, gateway_namespace = bootstrap_gateway_ref(
        aitenant_name=aitenant_preexist_test_params["aitenant_name"],
        aitenant_spec=aitenant_preexist_test_params["aitenant_spec"],
    )
    with bootstrap_gateway_context(
        admin_client=admin_client,
        gateway_name=gateway_name,
        gateway_namespace=gateway_namespace,
        teardown=teardown_resources,
    ) as gateway:
        yield gateway


@pytest.fixture
def aitenant_preexist_tenant_namespace(
    admin_client: DynamicClient,
    aitenant_preexist_test_params: AITenantTestParams,
    teardown_resources: bool,
) -> Generator[Namespace, Any, Any]:
    """Create the derived tenant namespace before AITenant reconciliation."""
    tenant_namespace_name = expected_tenant_namespace_name(
        aitenant_name=aitenant_preexist_test_params["aitenant_name"],
    )
    with Namespace(
        client=admin_client,
        name=tenant_namespace_name,
        teardown=teardown_resources,
    ) as tenant_namespace:
        yield tenant_namespace


@pytest.fixture
def aitenant_preexist(
    admin_client: DynamicClient,
    aitenant_infra_namespace: str,
    aitenant_preexist_test_params: AITenantTestParams,
    aitenant_preexist_bootstrap_gateway: Gateway,
    aitenant_preexist_tenant_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[AITenant, Any, Any]:
    """Deploy an AITenant CR that adopts a pre-existing derived tenant namespace."""
    with aitenant_from_spec(
        admin_client=admin_client,
        aitenant_name=aitenant_preexist_test_params["aitenant_name"],
        cr_namespace=aitenant_infra_namespace,
        aitenant_spec=aitenant_preexist_test_params["aitenant_spec"],
        teardown=teardown_resources,
    ) as aitenant:
        yield aitenant


@pytest.fixture
def ready_aitenant_preexist(aitenant_preexist: AITenant) -> Generator[AITenant, Any, Any]:
    """Wait until the pre-existing-namespace AITenant reports Ready."""
    deploy_and_verify_aitenant_ready(aitenant=aitenant_preexist)
    yield aitenant_preexist


@pytest.fixture
def aitenant_on_preexisting_derived_tenant_namespace(
    ready_aitenant_preexist: AITenant,
    aitenant_preexist_tenant_namespace: Namespace,
    aitenant_preexist_test_params: AITenantTestParams,
) -> AITenantPreexistingNamespaceContext:
    """Return context for a Ready AITenant that adopted a pre-existing tenant namespace."""
    tenant_namespace_name = expected_tenant_namespace_name(
        aitenant_name=aitenant_preexist_test_params["aitenant_name"],
    )
    return AITenantPreexistingNamespaceContext(
        aitenant=ready_aitenant_preexist,
        tenant_namespace=aitenant_preexist_tenant_namespace,
        tenant_namespace_name=tenant_namespace_name,
    )


@pytest.fixture
def aitenant_ns_clash_test_params() -> AITenantTestParams:
    """Return an AITenant name and spec for derived tenant namespace clash tests."""
    aitenant_name = f"e2e-aigw-ns-clash-{generate_random_name()}"
    return AITenantTestParams(
        aitenant_name=aitenant_name,
        aitenant_spec=build_aitenant_spec(aitenant_name=aitenant_name),
    )


@pytest.fixture
def aitenant_ns_clash_bootstrap_gateway(
    admin_client: DynamicClient,
    aitenant_ns_clash_test_params: AITenantTestParams,
    teardown_resources: bool,
) -> Generator[Gateway, Any, Any]:
    """Pre-provision the bootstrap Gateway for namespace clash tests."""
    gateway_name, gateway_namespace = bootstrap_gateway_ref(
        aitenant_name=aitenant_ns_clash_test_params["aitenant_name"],
        aitenant_spec=aitenant_ns_clash_test_params["aitenant_spec"],
    )
    with bootstrap_gateway_context(
        admin_client=admin_client,
        gateway_name=gateway_name,
        gateway_namespace=gateway_namespace,
        teardown=teardown_resources,
    ) as gateway:
        yield gateway


@pytest.fixture
def aitenant_ns_clash_tenant_namespace(
    admin_client: DynamicClient,
    aitenant_infra_namespace: str,
    aitenant_ns_clash_test_params: AITenantTestParams,
    teardown_resources: bool,
) -> Generator[Namespace, Any, Any]:
    """Create a derived tenant namespace already claimed by another AITenant."""
    tenant_namespace_name = expected_tenant_namespace_name(
        aitenant_name=aitenant_ns_clash_test_params["aitenant_name"],
    )
    with Namespace(
        client=admin_client,
        name=tenant_namespace_name,
        annotations={
            AIGATEWAY_NAME_ANNOTATION: "other-aitenant",
            AIGATEWAY_NAMESPACE_ANNOTATION: aitenant_infra_namespace,
        },
        teardown=teardown_resources,
    ) as tenant_namespace:
        yield tenant_namespace


@pytest.fixture
def aitenant_on_namespace_owned_by_other(
    admin_client: DynamicClient,
    aitenant_infra_namespace: str,
    aitenant_ns_clash_test_params: AITenantTestParams,
    aitenant_ns_clash_bootstrap_gateway: Gateway,
    aitenant_ns_clash_tenant_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[AITenant, Any, Any]:
    """Deploy an AITenant against a derived tenant namespace owned by another AITenant."""
    with aitenant_from_spec(
        admin_client=admin_client,
        aitenant_name=aitenant_ns_clash_test_params["aitenant_name"],
        cr_namespace=aitenant_infra_namespace,
        aitenant_spec=aitenant_ns_clash_test_params["aitenant_spec"],
        teardown=teardown_resources,
    ) as aitenant:
        yield aitenant


@pytest.fixture
def aitenant_outside_infra_namespace(
    request: pytest.FixtureRequest,
) -> tuple[str, str]:
    """Return AITenant name and invalid CR namespace for admission webhook rejection tests."""
    cr_namespace_key: str = request.param
    applications_namespace = py_config["applications_namespace"]
    namespace_map = {
        "applications_namespace": applications_namespace,
        "legacy_tenant_namespace": MAAS_SUBSCRIPTION_NAMESPACE,
    }
    aitenant_name = f"e2e-aigw-plc-{generate_random_name()}"
    return aitenant_name, namespace_map[cr_namespace_key]


@pytest.fixture
def aitenant_derived_test_params() -> AITenantTestParams:
    """Return an AITenant name and spec for derived tenant namespace verification."""
    aitenant_name = f"e2e-aigw-derive-{generate_random_name()}"
    return AITenantTestParams(
        aitenant_name=aitenant_name,
        aitenant_spec=build_aitenant_spec(aitenant_name=aitenant_name),
    )


@pytest.fixture
def aitenant_derived_bootstrap_gateway(
    admin_client: DynamicClient,
    aitenant_derived_test_params: AITenantTestParams,
    teardown_resources: bool,
) -> Generator[Gateway, Any, Any]:
    """Pre-provision the bootstrap Gateway for derived tenant namespace verification."""
    gateway_name, gateway_namespace = bootstrap_gateway_ref(
        aitenant_name=aitenant_derived_test_params["aitenant_name"],
        aitenant_spec=aitenant_derived_test_params["aitenant_spec"],
    )
    with bootstrap_gateway_context(
        admin_client=admin_client,
        gateway_name=gateway_name,
        gateway_namespace=gateway_namespace,
        teardown=teardown_resources,
    ) as gateway:
        yield gateway


@pytest.fixture
def aitenant_derived(
    admin_client: DynamicClient,
    aitenant_infra_namespace: str,
    aitenant_derived_test_params: AITenantTestParams,
    aitenant_derived_bootstrap_gateway: Gateway,
    teardown_resources: bool,
) -> Generator[AITenant, Any, Any]:
    """Deploy an AITenant CR for derived tenant namespace verification."""
    with aitenant_from_spec(
        admin_client=admin_client,
        aitenant_name=aitenant_derived_test_params["aitenant_name"],
        cr_namespace=aitenant_infra_namespace,
        aitenant_spec=aitenant_derived_test_params["aitenant_spec"],
        teardown=teardown_resources,
    ) as aitenant:
        yield aitenant


@pytest.fixture
def ready_aitenant_derived(aitenant_derived: AITenant) -> Generator[AITenant, Any, Any]:
    """Wait until the derived-namespace AITenant reports Ready."""
    deploy_and_verify_aitenant_ready(aitenant=aitenant_derived)
    yield aitenant_derived


@pytest.fixture
def aitenant_derived_namespace_case(
    ready_aitenant_derived: AITenant,
    aitenant_derived_test_params: AITenantTestParams,
) -> tuple[AITenantTestContext, str]:
    """Return Ready AITenant context and the expected derived tenant namespace name."""
    expected_tenant_namespace = expected_tenant_namespace_name(
        aitenant_name=aitenant_derived_test_params["aitenant_name"],
    )
    return build_aitenant_test_context(aitenant=ready_aitenant_derived), expected_tenant_namespace
