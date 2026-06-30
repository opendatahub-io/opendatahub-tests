from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.gateway_gateway_networking_k8s_io import Gateway
from ocp_resources.namespace import Namespace

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
from utilities.general import generate_random_name
from utilities.resources.aitenant import AITenant


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
