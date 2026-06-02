from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace

from tests.model_serving.maas_billing.multitenancy.aigateway.utils import (
    AIGATEWAY_INFRA_NAMESPACE,
    AIGatewayTestContext,
    build_aigateway_spec,
    build_aigateway_test_context,
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
        yield build_aigateway_test_context(aigateway=aigateway)
