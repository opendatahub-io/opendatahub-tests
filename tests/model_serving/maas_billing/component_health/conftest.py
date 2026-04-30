import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace

from utilities.resources.tenant import Tenant

TENANT_NAME = "default-tenant"


@pytest.fixture(scope="class")
def default_tenant(
    admin_client: DynamicClient,
    maas_subscription_namespace: Namespace,
) -> Tenant:
    """Return the default-tenant CR, asserting it exists."""
    tenant = Tenant(
        client=admin_client,
        name=TENANT_NAME,
        namespace=maas_subscription_namespace.name,
    )
    assert tenant.exists, (
        f"Tenant '{TENANT_NAME}' not found in namespace '{maas_subscription_namespace.name}'"
    )
    return tenant
