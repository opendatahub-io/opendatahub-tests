from collections.abc import Generator
from contextlib import ExitStack
from typing import Any

import pytest
import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.gateway_gateway_networking_k8s_io import Gateway
from ocp_resources.namespace import Namespace
from pytest_testconfig import config as py_config

from tests.model_serving.maas_billing.multitenancy.aitenant.utils import (
    AITENANT_INFRA_NAMESPACE,
    AITenantTestContext,
    aitenant_from_spec,
    bootstrap_gateway_context,
    bootstrap_gateway_ref,
    build_aitenant_spec,
    build_aitenant_test_context,
    deploy_and_verify_aitenant_ready,
)
from tests.model_serving.maas_billing.multitenancy.utils import (
    TENANT_ISOLATION_MODEL_NAME,
    TenantIsolationGovernance,
    gateway_ref_from_aitenant,
    isolation_bootstrap_gateway_context,
    isolation_tenant_api_key_id,
    isolation_tenant_api_key_plaintext,
    label_namespace_gateway_access,
    maas_api_base_url_for_gateway,
    make_tenant_model_accessible,
    provision_tenant_model,
    tenant_gateway_external_route,
    tenant_gateway_inference_url,
    tenant_isolation_auth_policy_name,
    tenant_isolation_subscription_name,
    verify_maas_api_deployment_for_aitenant,
    verify_maas_api_httproute_attached_to_gateway,
    verify_tenant_gateway_auth_policy_callback_url,
    wait_for_tenant_gateway_maas_api_reachable,
)
from utilities.general import generate_random_name
from utilities.resources.aitenant import AITenant
from utilities.resources.route import Route


@pytest.fixture(scope="session")
def aitenant_infra_namespace(admin_client: DynamicClient) -> str:
    """Return the infra namespace where AITenant objects are created."""
    infra_namespace = Namespace(client=admin_client, name=AITENANT_INFRA_NAMESPACE)
    assert infra_namespace.exists, (
        f"Infra namespace '{AITENANT_INFRA_NAMESPACE}' not found — required for AITenant multitenancy tests"
    )
    return AITENANT_INFRA_NAMESPACE


@pytest.fixture
def aitenant_test_params() -> dict[str, Any]:
    """Return the default AITenant name and spec for bootstrap tests."""
    aitenant_name = f"e2e-aigw-{generate_random_name()}"
    return {
        "aitenant_name": aitenant_name,
        "aitenant_spec": build_aitenant_spec(aitenant_name=aitenant_name),
    }


@pytest.fixture
def aitenant_bootstrap_gateway(
    admin_client: DynamicClient,
    aitenant_test_params: dict[str, Any],
    teardown_resources: bool,
) -> Generator[Gateway, Any, Any]:
    """Pre-provision the bootstrap Gateway required by an AITenant."""
    gateway_name, gateway_namespace = bootstrap_gateway_ref(
        aitenant_name=aitenant_test_params["aitenant_name"],
        aitenant_spec=aitenant_test_params["aitenant_spec"],
    )
    with bootstrap_gateway_context(
        admin_client=admin_client,
        gateway_name=gateway_name,
        gateway_namespace=gateway_namespace,
        teardown=teardown_resources,
    ) as gateway:
        yield gateway


@pytest.fixture
def aitenant(
    admin_client: DynamicClient,
    aitenant_infra_namespace: str,
    aitenant_test_params: dict[str, Any],
    aitenant_bootstrap_gateway: Gateway,
    teardown_resources: bool,
) -> Generator[AITenant, Any, Any]:
    """Deploy an AITenant CR after its bootstrap Gateway exists."""
    with aitenant_from_spec(
        admin_client=admin_client,
        aitenant_name=aitenant_test_params["aitenant_name"],
        cr_namespace=aitenant_infra_namespace,
        aitenant_spec=aitenant_test_params["aitenant_spec"],
        teardown=teardown_resources,
    ) as aitenant:
        yield aitenant


@pytest.fixture
def ready_aitenant(aitenant: AITenant) -> Generator[AITenant, Any, Any]:
    """Wait until the deployed AITenant reports Ready with phase Active."""
    deploy_and_verify_aitenant_ready(aitenant=aitenant)
    yield aitenant


@pytest.fixture
def aitenant_for_test(ready_aitenant: AITenant) -> AITenantTestContext:
    """Return bootstrap test context for a Ready AITenant."""
    return build_aitenant_test_context(aitenant=ready_aitenant)


@pytest.fixture(scope="class")
def isolation_primary_aitenant_test_params() -> dict[str, Any]:
    """Return the primary AITenant name and spec for auth isolation tests."""
    aitenant_name = f"e2e-mt-auth-a-{generate_random_name()}"
    return {
        "aitenant_name": aitenant_name,
        "aitenant_spec": build_aitenant_spec(aitenant_name=aitenant_name),
    }


@pytest.fixture(scope="class")
def isolation_primary_aitenant_bootstrap_gateway(
    admin_client: DynamicClient,
    isolation_primary_aitenant_test_params: dict[str, Any],
    teardown_resources: bool,
) -> Generator[Gateway, Any, Any]:
    """Pre-provision the bootstrap Gateway for the primary auth isolation AITenant."""
    gateway_name, gateway_namespace = bootstrap_gateway_ref(
        aitenant_name=isolation_primary_aitenant_test_params["aitenant_name"],
        aitenant_spec=isolation_primary_aitenant_test_params["aitenant_spec"],
    )
    applications_namespace = py_config["applications_namespace"]
    with isolation_bootstrap_gateway_context(
        admin_client=admin_client,
        gateway_name=gateway_name,
        applications_namespace=applications_namespace,
        gateway_namespace=gateway_namespace,
        teardown=teardown_resources,
    ) as gateway:
        yield gateway


@pytest.fixture(scope="class")
def isolation_primary_gateway_route(
    admin_client: DynamicClient,
    maas_host: str,
    isolation_primary_aitenant_bootstrap_gateway: Gateway,
    teardown_resources: bool,
) -> Generator[Route, Any, Any]:
    """Expose the primary tenant Gateway on an external hostname."""
    with tenant_gateway_external_route(
        admin_client=admin_client,
        gateway_name=isolation_primary_aitenant_bootstrap_gateway.name,
        maas_host=maas_host,
        teardown=teardown_resources,
    ) as route:
        yield route


@pytest.fixture(scope="class")
def tenant_a_base_url(
    maas_host: str,
    isolation_primary_aitenant_bootstrap_gateway: Gateway,
    isolation_primary_gateway_route: Route,
) -> str:
    """Return the maas-api base URL routed through tenant A's Gateway."""
    _ = isolation_primary_gateway_route
    return maas_api_base_url_for_gateway(
        gateway_name=isolation_primary_aitenant_bootstrap_gateway.name,
        maas_host=maas_host,
        scheme="https",
    )


@pytest.fixture(scope="class")
def isolation_primary_aitenant(
    admin_client: DynamicClient,
    aitenant_infra_namespace: str,
    isolation_primary_aitenant_test_params: dict[str, Any],
    isolation_primary_aitenant_bootstrap_gateway: Gateway,
    teardown_resources: bool,
) -> Generator[AITenant, Any, Any]:
    """Deploy the primary AITenant CR for auth isolation tests."""
    with aitenant_from_spec(
        admin_client=admin_client,
        aitenant_name=isolation_primary_aitenant_test_params["aitenant_name"],
        cr_namespace=aitenant_infra_namespace,
        aitenant_spec=isolation_primary_aitenant_test_params["aitenant_spec"],
        teardown=teardown_resources,
    ) as aitenant:
        yield aitenant


@pytest.fixture(scope="class")
def isolation_primary_ready_aitenant(
    isolation_primary_aitenant: AITenant,
) -> Generator[AITenant, Any, Any]:
    """Wait until the primary auth isolation AITenant reports Ready with phase Active."""
    deploy_and_verify_aitenant_ready(aitenant=isolation_primary_aitenant)
    yield isolation_primary_aitenant


@pytest.fixture(scope="class")
def isolation_primary_aitenant_for_test(
    isolation_primary_ready_aitenant: AITenant,
) -> AITenantTestContext:
    """Return test context for the primary Ready AITenant."""
    return build_aitenant_test_context(aitenant=isolation_primary_ready_aitenant)


@pytest.fixture(scope="class")
def isolation_secondary_aitenant_test_params() -> dict[str, Any]:
    """Return the secondary AITenant name and spec for auth isolation tests."""
    aitenant_name = f"e2e-mt-auth-b-{generate_random_name()}"
    return {
        "aitenant_name": aitenant_name,
        "aitenant_spec": build_aitenant_spec(aitenant_name=aitenant_name),
    }


@pytest.fixture(scope="class")
def isolation_secondary_aitenant_bootstrap_gateway(
    admin_client: DynamicClient,
    isolation_secondary_aitenant_test_params: dict[str, Any],
    teardown_resources: bool,
) -> Generator[Gateway, Any, Any]:
    """Pre-provision the bootstrap Gateway for the secondary auth isolation AITenant."""
    gateway_name, gateway_namespace = bootstrap_gateway_ref(
        aitenant_name=isolation_secondary_aitenant_test_params["aitenant_name"],
        aitenant_spec=isolation_secondary_aitenant_test_params["aitenant_spec"],
    )
    applications_namespace = py_config["applications_namespace"]
    with isolation_bootstrap_gateway_context(
        admin_client=admin_client,
        gateway_name=gateway_name,
        applications_namespace=applications_namespace,
        gateway_namespace=gateway_namespace,
        teardown=teardown_resources,
    ) as gateway:
        yield gateway


@pytest.fixture(scope="class")
def isolation_secondary_gateway_route(
    admin_client: DynamicClient,
    maas_host: str,
    isolation_secondary_aitenant_bootstrap_gateway: Gateway,
    teardown_resources: bool,
) -> Generator[Route, Any, Any]:
    """Expose the secondary tenant Gateway on an external hostname."""
    with tenant_gateway_external_route(
        admin_client=admin_client,
        gateway_name=isolation_secondary_aitenant_bootstrap_gateway.name,
        maas_host=maas_host,
        teardown=teardown_resources,
    ) as route:
        yield route


@pytest.fixture(scope="class")
def tenant_b_base_url(
    maas_host: str,
    isolation_secondary_aitenant_bootstrap_gateway: Gateway,
    isolation_secondary_gateway_route: Route,
) -> str:
    """Return the maas-api base URL routed through tenant B's Gateway."""
    _ = isolation_secondary_gateway_route
    return maas_api_base_url_for_gateway(
        gateway_name=isolation_secondary_aitenant_bootstrap_gateway.name,
        maas_host=maas_host,
        scheme="https",
    )


@pytest.fixture(scope="class")
def isolation_secondary_aitenant(
    admin_client: DynamicClient,
    aitenant_infra_namespace: str,
    isolation_secondary_aitenant_test_params: dict[str, Any],
    isolation_secondary_aitenant_bootstrap_gateway: Gateway,
    teardown_resources: bool,
) -> Generator[AITenant, Any, Any]:
    """Deploy the secondary AITenant CR for auth isolation tests."""
    with aitenant_from_spec(
        admin_client=admin_client,
        aitenant_name=isolation_secondary_aitenant_test_params["aitenant_name"],
        cr_namespace=aitenant_infra_namespace,
        aitenant_spec=isolation_secondary_aitenant_test_params["aitenant_spec"],
        teardown=teardown_resources,
    ) as aitenant:
        yield aitenant


@pytest.fixture(scope="class")
def isolation_secondary_ready_aitenant(
    isolation_secondary_aitenant: AITenant,
) -> Generator[AITenant, Any, Any]:
    """Wait until the secondary auth isolation AITenant reports Ready with phase Active."""
    deploy_and_verify_aitenant_ready(aitenant=isolation_secondary_aitenant)
    yield isolation_secondary_aitenant


@pytest.fixture(scope="class")
def isolation_secondary_aitenant_for_test(
    isolation_secondary_ready_aitenant: AITenant,
) -> AITenantTestContext:
    """Return test context for the secondary Ready AITenant."""
    return build_aitenant_test_context(aitenant=isolation_secondary_ready_aitenant)


@pytest.fixture(scope="class")
def two_aitenant_test_contexts(
    isolation_primary_aitenant_for_test: AITenantTestContext,
    isolation_secondary_aitenant_for_test: AITenantTestContext,
) -> tuple[AITenantTestContext, AITenantTestContext]:
    """Return Ready AITenant contexts for a two-tenant auth isolation scenario."""
    return isolation_primary_aitenant_for_test, isolation_secondary_aitenant_for_test


@pytest.fixture(scope="class")
def isolation_tenant_governance(
    admin_client: DynamicClient,
    two_aitenant_test_contexts: tuple[AITenantTestContext, AITenantTestContext],
    maas_free_group: str,
    teardown_resources: bool,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
) -> Generator[list[TenantIsolationGovernance], Any, Any]:
    """Provision a tenant-local model, auth policy, and subscription per AITenant (dev order)."""
    governance_contexts: list[TenantIsolationGovernance] = []
    with ExitStack() as stack:
        for test_context in two_aitenant_test_contexts:
            gateway_name, gateway_namespace = gateway_ref_from_aitenant(aitenant=test_context["aitenant"])
            tenant_namespace_name = test_context["tenant_namespace_name"]
            aitenant_name = test_context["aitenant_name"]
            auth_policy_name = tenant_isolation_auth_policy_name(aitenant_name=aitenant_name)
            subscription_name = tenant_isolation_subscription_name(aitenant_name=aitenant_name)

            stack.enter_context(
                cm=provision_tenant_model(
                    admin_client=admin_client,
                    model_name=TENANT_ISOLATION_MODEL_NAME,
                    tenant_namespace_name=tenant_namespace_name,
                    gateway_name=gateway_name,
                    gateway_namespace=gateway_namespace,
                    aws_access_key=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    aws_s3_region=models_s3_bucket_region,
                    aws_s3_bucket=models_s3_bucket_name,
                    aws_s3_endpoint=models_s3_bucket_endpoint,
                    teardown=teardown_resources,
                )
            )
            stack.enter_context(
                cm=make_tenant_model_accessible(
                    admin_client=admin_client,
                    model_name=TENANT_ISOLATION_MODEL_NAME,
                    model_namespace=tenant_namespace_name,
                    tenant_namespace_name=tenant_namespace_name,
                    auth_policy_name=auth_policy_name,
                    subscription_name=subscription_name,
                    owner_group_name="system:authenticated",
                    free_group_name=maas_free_group,
                    teardown=teardown_resources,
                )
            )
            governance_contexts.append(
                TenantIsolationGovernance(
                    aitenant_name=aitenant_name,
                    tenant_namespace_name=tenant_namespace_name,
                    model_name=TENANT_ISOLATION_MODEL_NAME,
                    model_namespace=tenant_namespace_name,
                    auth_policy_name=auth_policy_name,
                    subscription_name=subscription_name,
                )
            )
        yield governance_contexts


@pytest.fixture(scope="class")
def tenant_a_subscription_name(
    isolation_tenant_governance: list[TenantIsolationGovernance],
) -> str:
    """Return the MaaSSubscription name for tenant A."""
    return isolation_tenant_governance[0]["subscription_name"]


@pytest.fixture(scope="class")
def tenant_b_subscription_name(
    isolation_tenant_governance: list[TenantIsolationGovernance],
) -> str:
    """Return the MaaSSubscription name for tenant B."""
    return isolation_tenant_governance[1]["subscription_name"]


@pytest.fixture(scope="class")
def per_tenant_maas_api_ready(
    admin_client: DynamicClient,
    two_aitenant_test_contexts: tuple[AITenantTestContext, AITenantTestContext],
    isolation_tenant_governance: list[TenantIsolationGovernance],
    isolation_primary_gateway_route: Route,
    isolation_secondary_gateway_route: Route,
    maas_host: str,
    request_session_http: requests.Session,
) -> None:
    """Wait until per-tenant maas-api is cluster-ready and reachable via external Routes."""
    _ = isolation_tenant_governance
    _ = isolation_primary_gateway_route
    _ = isolation_secondary_gateway_route
    applications_namespace = py_config["applications_namespace"]
    for test_context in two_aitenant_test_contexts:
        gateway_name, gateway_namespace = gateway_ref_from_aitenant(aitenant=test_context["aitenant"])
        label_namespace_gateway_access(
            admin_client=admin_client,
            namespace_name=test_context["tenant_namespace_name"],
            gateway_name=gateway_name,
        )
        verify_maas_api_deployment_for_aitenant(
            admin_client=admin_client,
            applications_namespace=applications_namespace,
            aitenant_name=test_context["aitenant_name"],
            tenant_namespace_name=test_context["tenant_namespace_name"],
        )
        verify_maas_api_httproute_attached_to_gateway(
            admin_client=admin_client,
            applications_namespace=applications_namespace,
            aitenant_name=test_context["aitenant_name"],
            tenant_namespace_name=test_context["tenant_namespace_name"],
            gateway_name=gateway_name,
            gateway_namespace=gateway_namespace,
        )
        verify_tenant_gateway_auth_policy_callback_url(
            admin_client=admin_client,
            gateway_name=gateway_name,
            gateway_namespace=gateway_namespace,
            aitenant_name=test_context["aitenant_name"],
            applications_namespace=applications_namespace,
        )
        wait_for_tenant_gateway_maas_api_reachable(
            request_session_http=request_session_http,
            gateway_name=gateway_name,
            maas_host=maas_host,
        )


@pytest.fixture(scope="class")
def tenant_a_api_key_id(
    per_tenant_maas_api_ready: None,
    request_session_http: requests.Session,
    tenant_a_base_url: str,
    tenant_a_subscription_name: str,
    current_client_token: str,
) -> Generator[str, Any, Any]:
    """Create an API key in tenant A and revoke it after the test class completes."""
    with isolation_tenant_api_key_id(
        request_session_http=request_session_http,
        base_url=tenant_a_base_url,
        ocp_user_token=current_client_token,
        subscription_name=tenant_a_subscription_name,
        key_name_prefix="e2e-mt-isolation-a",
        fixture_label="tenant_a_api_key_id",
    ) as key_id:
        yield key_id


@pytest.fixture(scope="class")
def tenant_b_api_key_id(
    per_tenant_maas_api_ready: None,
    request_session_http: requests.Session,
    tenant_b_base_url: str,
    tenant_b_subscription_name: str,
    current_client_token: str,
) -> Generator[str, Any, Any]:
    """Create an API key in tenant B and revoke it after the test class completes."""
    with isolation_tenant_api_key_id(
        request_session_http=request_session_http,
        base_url=tenant_b_base_url,
        ocp_user_token=current_client_token,
        subscription_name=tenant_b_subscription_name,
        key_name_prefix="e2e-mt-isolation-b",
        fixture_label="tenant_b_api_key_id",
    ) as key_id:
        yield key_id


@pytest.fixture(scope="class")
def tenant_inference_chat_payload() -> dict[str, Any]:
    """Return the chat completions payload for tenant inference tests."""
    return {
        "model": TENANT_ISOLATION_MODEL_NAME,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 16,
    }


@pytest.fixture(scope="class")
def tenant_a_inference_url(
    maas_host: str,
    isolation_primary_aitenant_for_test: AITenantTestContext,
    isolation_primary_aitenant_bootstrap_gateway: Gateway,
) -> str:
    """Return tenant A external inference URL for the tenant-local model."""
    return tenant_gateway_inference_url(
        gateway_name=isolation_primary_aitenant_bootstrap_gateway.name,
        maas_host=maas_host,
        tenant_namespace_name=isolation_primary_aitenant_for_test["tenant_namespace_name"],
        model_name=TENANT_ISOLATION_MODEL_NAME,
    )


@pytest.fixture(scope="class")
def tenant_b_inference_url(
    maas_host: str,
    isolation_secondary_aitenant_for_test: AITenantTestContext,
    isolation_secondary_aitenant_bootstrap_gateway: Gateway,
) -> str:
    """Return tenant B external inference URL for the tenant-local model."""
    return tenant_gateway_inference_url(
        gateway_name=isolation_secondary_aitenant_bootstrap_gateway.name,
        maas_host=maas_host,
        tenant_namespace_name=isolation_secondary_aitenant_for_test["tenant_namespace_name"],
        model_name=TENANT_ISOLATION_MODEL_NAME,
    )


@pytest.fixture(scope="class")
def tenant_a_inference_headers(
    per_tenant_maas_api_ready: None,
    request_session_http: requests.Session,
    tenant_a_base_url: str,
    tenant_a_subscription_name: str,
    current_client_token: str,
) -> Generator[dict[str, str], Any, Any]:
    """Create a tenant A API key and return Authorization headers for inference."""
    with isolation_tenant_api_key_plaintext(
        request_session_http=request_session_http,
        base_url=tenant_a_base_url,
        ocp_user_token=current_client_token,
        subscription_name=tenant_a_subscription_name,
        key_name_prefix="e2e-mt-inference-a",
        fixture_label="tenant_a_inference_headers",
    ) as plaintext_key:
        yield {"Authorization": f"Bearer {plaintext_key}"}
