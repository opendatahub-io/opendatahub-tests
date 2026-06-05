"""Fixtures for body-based routing (BBR) integration tests."""

from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace

from tests.model_serving.maas_billing.body_base_routing.pre_auth_model_header.utils import (
    BBR_PRE_PROCESSING_DEPLOYMENT_NAME,
    get_bbr_envoy_filter_config_patches,
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


@pytest.fixture(scope="class")
def bbr_envoy_filter_config_patches(
    admin_client: DynamicClient,
    bbr_gateway_namespace: str,
) -> list[Any]:
    """Return BBR EnvoyFilter configPatches once per test class to avoid redundant API calls."""
    return get_bbr_envoy_filter_config_patches(
        admin_client=admin_client,
        gateway_namespace=bbr_gateway_namespace,
    )
