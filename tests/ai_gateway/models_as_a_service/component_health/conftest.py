import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace

from utilities.resources.maastenantconfig import MaasTenantConfig

MAAS_TENANT_CONFIG_NAME = "default-tenant"


@pytest.fixture(scope="class")
def default_maas_tenant_config(
    admin_client: DynamicClient,
    maas_subscription_namespace: Namespace,
) -> MaasTenantConfig:
    """Return the default MaasTenantConfig CR, asserting it exists."""
    maas_tenant_config = MaasTenantConfig(
        client=admin_client,
        name=MAAS_TENANT_CONFIG_NAME,
        namespace=maas_subscription_namespace.name,
    )
    assert maas_tenant_config.exists, (
        f"MaasTenantConfig '{MAAS_TENANT_CONFIG_NAME}' not found in namespace '{maas_subscription_namespace.name}'"
    )
    return maas_tenant_config
