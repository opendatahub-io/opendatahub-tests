"""Fixtures for body-based routing (BBR) integration tests."""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace

from tests.model_serving.maas_billing.body_base_routing.pre_auth_model_header.utils import (
    BBR_PRE_PROCESSING_DEPLOYMENT_NAME,
)
from utilities.constants import MAAS_GATEWAY_NAMESPACE


@pytest.fixture(scope="session")
def bbr_gateway_namespace(admin_client: DynamicClient) -> str:
    """Return the gateway namespace where BBR infrastructure is deployed."""
    gateway_namespace = Namespace(client=admin_client, name=MAAS_GATEWAY_NAMESPACE)
    assert gateway_namespace.exists, (
        f"Gateway namespace '{MAAS_GATEWAY_NAMESPACE}' not found — "
        f"required for BBR tests ({BBR_PRE_PROCESSING_DEPLOYMENT_NAME} is deployed here)"
    )
    return MAAS_GATEWAY_NAMESPACE
