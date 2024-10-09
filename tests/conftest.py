import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.resource import get_client


@pytest.fixture(scope="session")
def admin_client() -> DynamicClient:
    return get_client()


@pytest.fixture(scope="class")
def model_namespace(request, admin_client: DynamicClient) -> Namespace:
    with Namespace(
        client=admin_client,
        name=request.param["name"],
    ) as ns:
        ns.wait_for_status(status=Namespace.Status.ACTIVE, timeout=120)
        yield ns
