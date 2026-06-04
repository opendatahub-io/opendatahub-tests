from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.gateway_gateway_networking_k8s_io import Gateway
from ocp_resources.namespace import Namespace
from pytest_testconfig import config as py_config

from tests.model_serving.maas_billing.maas_subscription.utils import MAAS_SUBSCRIPTION_NAMESPACE
from tests.model_serving.maas_billing.multitenancy.aigateway.utils import (
    AIGATEWAY_INFRA_NAMESPACE,
    AIGATEWAY_NAME_ANNOTATION,
    AIGATEWAY_NAMESPACE_ANNOTATION,
    AIGATEWAY_TEST_OIDC_SPEC,
    AIGATEWAY_TEST_RBAC_ADMINS,
    AIGatewayPreexistingNamespaceContext,
    AIGatewayTestContext,
    aigateway_from_spec,
    build_aigateway_spec,
    build_aigateway_test_context,
    deploy_and_verify_aigateway_ready,
    tenant_namespace_name_for_aigateway,
)
from utilities.constants import MAAS_GATEWAY_NAMESPACE
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


@pytest.fixture
def ready_aigateway_with_cleanup_on_delete(
    admin_client: DynamicClient,
    aigateway_infra_namespace: str,
    teardown_resources: bool,
) -> Generator[AIGatewayTestContext, Any, Any]:
    """Create a Ready AIGateway with cleanupOnDelete=true; test owns deletion."""
    aigateway_name = f"e2e-aigw-del-{generate_random_name()}"
    aigateway_spec = build_aigateway_spec(aigateway_name=aigateway_name, cleanup_on_delete=True)
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
def ready_aigateway_without_cleanup_on_delete(
    admin_client: DynamicClient,
    aigateway_infra_namespace: str,
    teardown_resources: bool,
) -> Generator[AIGatewayTestContext, Any, Any]:
    """Create a Ready AIGateway with cleanupOnDelete=false; test owns deletion."""
    aigateway_name = f"e2e-aigw-keep-{generate_random_name()}"
    tenant_namespace_name = tenant_namespace_name_for_aigateway(aigateway_name=aigateway_name)
    with Namespace(client=admin_client, name=tenant_namespace_name, teardown=teardown_resources):
        aigateway_spec = build_aigateway_spec(
            aigateway_name=aigateway_name,
            tenant_namespace_name=tenant_namespace_name,
            cleanup_on_delete=False,
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
def aigateway_on_labeled_preexisting_namespace(
    admin_client: DynamicClient,
    aigateway_infra_namespace: str,
    teardown_resources: bool,
) -> Generator[AIGatewayPreexistingNamespaceContext, Any, Any]:
    """Create a pre-existing labeled tenant namespace and a Ready adopting AIGateway."""
    aigateway_name = f"e2e-aigw-preexist-{generate_random_name()}"
    tenant_namespace_name = tenant_namespace_name_for_aigateway(aigateway_name=aigateway_name)
    with Namespace(
        client=admin_client,
        name=tenant_namespace_name,
        annotations={
            AIGATEWAY_NAME_ANNOTATION: aigateway_name,
            AIGATEWAY_NAMESPACE_ANNOTATION: aigateway_infra_namespace,
        },
        teardown=teardown_resources,
    ) as tenant_namespace:
        aigateway_spec = build_aigateway_spec(
            aigateway_name=aigateway_name,
            tenant_namespace_name=tenant_namespace_name,
            cleanup_on_delete=True,
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
            yield AIGatewayPreexistingNamespaceContext(
                aigateway=aigateway,
                tenant_namespace=tenant_namespace,
                tenant_namespace_name=tenant_namespace_name,
            )


@pytest.fixture
def aigateway_pending_missing_tenant_namespace(
    admin_client: DynamicClient,
    aigateway_infra_namespace: str,
    teardown_resources: bool,
) -> Generator[AIGateway, Any, Any]:
    """Create an AIGateway with create=false against a missing tenant namespace."""
    aigateway_name = f"e2e-aigw-pend-{generate_random_name()}"
    tenant_namespace_name = tenant_namespace_name_for_aigateway(aigateway_name=aigateway_name)
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
        yield aigateway


@pytest.fixture
def aigateway_on_namespace_owned_by_other(
    admin_client: DynamicClient,
    aigateway_infra_namespace: str,
    teardown_resources: bool,
) -> Generator[AIGateway, Any, Any]:
    """Create an AIGateway against a namespace already claimed by another AIGateway."""
    aigateway_name = f"e2e-aigw-ns-clash-{generate_random_name()}"
    shared_namespace_name = f"e2e-aigw-shared-{generate_random_name()}"
    with Namespace(
        client=admin_client,
        name=shared_namespace_name,
        annotations={
            AIGATEWAY_NAME_ANNOTATION: "other-aigw",
            AIGATEWAY_NAMESPACE_ANNOTATION: aigateway_infra_namespace,
        },
        teardown=teardown_resources,
    ):
        aigateway_spec = build_aigateway_spec(
            aigateway_name=aigateway_name,
            tenant_namespace_name=shared_namespace_name,
            create_tenant_namespace=False,
        )
        with aigateway_from_spec(
            admin_client=admin_client,
            aigateway_name=aigateway_name,
            cr_namespace=aigateway_infra_namespace,
            aigateway_spec=aigateway_spec,
            teardown=teardown_resources,
        ) as aigateway:
            yield aigateway


@pytest.fixture
def aigateway_on_gateway_owned_by_other(
    admin_client: DynamicClient,
    aigateway_infra_namespace: str,
    teardown_resources: bool,
) -> Generator[AIGateway, Any, Any]:
    """Create an AIGateway against a Gateway already claimed by another AIGateway."""
    aigateway_name = f"e2e-aigw-gw-clash-{generate_random_name()}"
    contested_gateway_name = f"e2e-aigw-contested-{generate_random_name()}"
    tenant_namespace_name = tenant_namespace_name_for_aigateway(aigateway_name=aigateway_name)
    with (
        Gateway(
            client=admin_client,
            name=contested_gateway_name,
            namespace=MAAS_GATEWAY_NAMESPACE,
            gateway_class_name="openshift-default",
            listeners=[{"name": "http", "port": 80, "protocol": "HTTP"}],
            annotations={
                AIGATEWAY_NAME_ANNOTATION: "other-aigw",
                AIGATEWAY_NAMESPACE_ANNOTATION: aigateway_infra_namespace,
            },
            teardown=teardown_resources,
        ),
        Namespace(client=admin_client, name=tenant_namespace_name, teardown=teardown_resources),
    ):
        aigateway_spec = build_aigateway_spec(
            aigateway_name=aigateway_name,
            tenant_namespace_name=tenant_namespace_name,
            gateway_name=contested_gateway_name,
            create_tenant_namespace=False,
        )
        with aigateway_from_spec(
            admin_client=admin_client,
            aigateway_name=aigateway_name,
            cr_namespace=aigateway_infra_namespace,
            aigateway_spec=aigateway_spec,
            teardown=teardown_resources,
        ) as aigateway:
            yield aigateway


@pytest.fixture
def invalid_placement_aigateway(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    aigateway_infra_namespace: str,
    teardown_resources: bool,
) -> Generator[AIGateway, Any, Any]:
    """Create an AIGateway in an invalid namespace placement scenario."""
    _test_id, cr_namespace, tenant_namespace_name = request.param
    namespace_map = {
        "applications_namespace": py_config["applications_namespace"],
        "legacy_tenant_namespace": MAAS_SUBSCRIPTION_NAMESPACE,
        "infra_namespace": aigateway_infra_namespace,
    }
    resolved_cr_namespace = namespace_map[cr_namespace]
    aigateway_name = f"e2e-aigw-plc-{generate_random_name()}"
    resolved_tenant_namespace = (
        namespace_map[tenant_namespace_name]
        if tenant_namespace_name is not None
        else tenant_namespace_name_for_aigateway(aigateway_name=aigateway_name)
    )
    aigateway_spec = build_aigateway_spec(
        aigateway_name=aigateway_name,
        tenant_namespace_name=resolved_tenant_namespace,
    )
    with aigateway_from_spec(
        admin_client=admin_client,
        aigateway_name=aigateway_name,
        cr_namespace=resolved_cr_namespace,
        aigateway_spec=aigateway_spec,
        teardown=teardown_resources,
    ) as aigateway:
        yield aigateway


@pytest.fixture
def aigateway_deploy_tls_without_domain(
    admin_client: DynamicClient,
    aigateway_infra_namespace: str,
    teardown_resources: bool,
) -> Generator[AIGateway, Any, Any]:
    """Yield an undeployed AIGateway spec with tls but no domain for API validation tests."""
    aigateway_name = f"e2e-aigw-cel-{generate_random_name()}"
    aigateway_spec = build_aigateway_spec(aigateway_name=aigateway_name)
    aigateway_spec["tls"] = {"certificateRef": {"name": "test-tls-cert"}}
    with aigateway_from_spec(
        admin_client=admin_client,
        aigateway_name=aigateway_name,
        cr_namespace=aigateway_infra_namespace,
        aigateway_spec=aigateway_spec,
        teardown=teardown_resources,
    ) as aigateway:
        yield aigateway
