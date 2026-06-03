from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace

from tests.model_serving.maas_billing.multitenancy.aigateway.utils import (
    AIGATEWAY_INFRA_NAMESPACE,
    AIGATEWAY_TEST_OIDC_SPEC,
    AIGATEWAY_TEST_RBAC_ADMINS,
    AIGatewayTestContext,
    aigateway_from_spec,
    build_aigateway_spec,
    build_aigateway_test_context,
    deploy_and_verify_aigateway_ready,
    tenant_namespace_name_for_aigateway,
)
from utilities.general import generate_random_name
from utilities.resources.aigateway import AIGateway


@pytest.fixture(scope="session")
def aigateway_infra_namespace(admin_client: DynamicClient) -> str:
    """Return the infra namespace where AIGateway objects are created."""
    infra_namespace = Namespace(client=admin_client, name=AIGATEWAY_INFRA_NAMESPACE)
    assert infra_namespace.exists, (
        f"Infra namespace '{AIGATEWAY_INFRA_NAMESPACE}' not found — required for AIGateway multitenancy tests"
    )
    return AIGATEWAY_INFRA_NAMESPACE


@pytest.fixture
def aigateway_for_test(
    admin_client: DynamicClient,
    aigateway_infra_namespace: str,
    teardown_resources: bool,
) -> Generator[AIGatewayTestContext, Any, Any]:
    """Create a disposable AIGateway and yield context for bootstrap assertions."""
    aigateway_name = f"e2e-aigw-{generate_random_name()}"
    aigateway_spec = build_aigateway_spec(aigateway_name=aigateway_name)
    with AIGateway(
        client=admin_client,
        name=aigateway_name,
        namespace=aigateway_infra_namespace,
        tenant_namespace=aigateway_spec["tenantNamespace"],
        gateway=aigateway_spec["gateway"],
        teardown=teardown_resources,
        wait_for_resource=True,
    ) as aigateway:
        deploy_and_verify_aigateway_ready(aigateway=aigateway)
        yield build_aigateway_test_context(aigateway=aigateway)


@pytest.fixture
def aigateway_adopting_preexisting_namespace(
    admin_client: DynamicClient,
    aigateway_infra_namespace: str,
    teardown_resources: bool,
) -> Generator[AIGatewayTestContext, Any, Any]:
    """Create a tenant namespace first, then an AIGateway that adopts it with create=false."""
    aigateway_name = f"e2e-aigw-adopt-ns-{generate_random_name()}"
    tenant_namespace_name = tenant_namespace_name_for_aigateway(aigateway_name=aigateway_name)
    with Namespace(client=admin_client, name=tenant_namespace_name, teardown=teardown_resources):
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
            yield build_aigateway_test_context(aigateway=aigateway)


@pytest.fixture
def aigateway_with_domain(
    admin_client: DynamicClient,
    aigateway_infra_namespace: str,
    teardown_resources: bool,
) -> Generator[AIGatewayTestContext, Any, Any]:
    """Create an AIGateway with spec.domain configured."""
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
        yield build_aigateway_test_context(aigateway=aigateway)


@pytest.fixture
def aigateway_with_tls(
    admin_client: DynamicClient,
    aigateway_infra_namespace: str,
    teardown_resources: bool,
) -> Generator[AIGatewayTestContext, Any, Any]:
    """Create an AIGateway with spec.domain and spec.tls configured."""
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
        yield build_aigateway_test_context(aigateway=aigateway)


@pytest.fixture
def aigateway_with_oidc(
    admin_client: DynamicClient,
    aigateway_infra_namespace: str,
    teardown_resources: bool,
) -> Generator[AIGatewayTestContext, Any, Any]:
    """Create an AIGateway with spec.oidc configured."""
    aigateway_name = f"e2e-aigw-oidc-{generate_random_name()}"
    aigateway_spec = build_aigateway_spec(aigateway_name=aigateway_name, oidc=AIGATEWAY_TEST_OIDC_SPEC)
    with aigateway_from_spec(
        admin_client=admin_client,
        aigateway_name=aigateway_name,
        cr_namespace=aigateway_infra_namespace,
        aigateway_spec=aigateway_spec,
        teardown=teardown_resources,
    ) as aigateway:
        deploy_and_verify_aigateway_ready(aigateway=aigateway)
        yield build_aigateway_test_context(aigateway=aigateway)


@pytest.fixture
def aigateway_with_rbac_admins(
    admin_client: DynamicClient,
    aigateway_infra_namespace: str,
    teardown_resources: bool,
) -> Generator[AIGatewayTestContext, Any, Any]:
    """Create an AIGateway with spec.rbac.admins configured."""
    aigateway_name = f"e2e-aigw-rbac-{generate_random_name()}"
    aigateway_spec = build_aigateway_spec(
        aigateway_name=aigateway_name,
        rbac_admins=AIGATEWAY_TEST_RBAC_ADMINS,
    )
    with aigateway_from_spec(
        admin_client=admin_client,
        aigateway_name=aigateway_name,
        cr_namespace=aigateway_infra_namespace,
        aigateway_spec=aigateway_spec,
        teardown=teardown_resources,
    ) as aigateway:
        deploy_and_verify_aigateway_ready(aigateway=aigateway)
        yield build_aigateway_test_context(aigateway=aigateway)


@pytest.fixture
def aigateway_without_rbac_admins(
    admin_client: DynamicClient,
    aigateway_infra_namespace: str,
    teardown_resources: bool,
) -> Generator[AIGatewayTestContext, Any, Any]:
    """Create an AIGateway without spec.rbac.admins configured."""
    aigateway_name = f"e2e-aigw-no-rbac-{generate_random_name()}"
    aigateway_spec = build_aigateway_spec(aigateway_name=aigateway_name)
    with aigateway_from_spec(
        admin_client=admin_client,
        aigateway_name=aigateway_name,
        cr_namespace=aigateway_infra_namespace,
        aigateway_spec=aigateway_spec,
        teardown=teardown_resources,
    ) as aigateway:
        deploy_and_verify_aigateway_ready(aigateway=aigateway)
        yield build_aigateway_test_context(aigateway=aigateway)
