import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.network_policy import NetworkPolicy
from pytest_testconfig import config as py_config

from tests.ai_safety.trustyai_operator.constants import (
    TRUSTYAI_OPERATOR_DEPLOYMENT,
    TRUSTYAI_OPERATOR_EGRESS_NETWORK_POLICY,
    TRUSTYAI_OPERATOR_INGRESS_NETWORK_POLICY,
)


@pytest.fixture(scope="session")
def trustyai_operator_installed(admin_client: DynamicClient) -> None:
    """Skip when the TrustyAI workload operator is not deployed in the applications namespace."""
    operator = Deployment(
        client=admin_client,
        name=TRUSTYAI_OPERATOR_DEPLOYMENT,
        namespace=py_config["applications_namespace"],
    )
    if not operator.exists:
        pytest.skip(
            f"Deployment {TRUSTYAI_OPERATOR_DEPLOYMENT} is not present in "
            f"{py_config['applications_namespace']}; TrustyAI is not installed."
        )


@pytest.fixture(scope="session")
def trustyai_operator_ingress_network_policy(
    admin_client: DynamicClient,
    trustyai_operator_installed: None,
) -> NetworkPolicy:
    """Wrap the ingress NetworkPolicy admitted for the workload operator manager pod."""
    return NetworkPolicy(
        client=admin_client,
        name=TRUSTYAI_OPERATOR_INGRESS_NETWORK_POLICY,
        namespace=py_config["applications_namespace"],
        ensure_exists=True,
    )


@pytest.fixture(scope="session")
def trustyai_operator_egress_network_policy(
    admin_client: DynamicClient,
    trustyai_operator_installed: None,
) -> NetworkPolicy:
    """Wrap the egress NetworkPolicy declared for the workload operator manager pod."""
    return NetworkPolicy(
        client=admin_client,
        name=TRUSTYAI_OPERATOR_EGRESS_NETWORK_POLICY,
        namespace=py_config["applications_namespace"],
        ensure_exists=True,
    )
