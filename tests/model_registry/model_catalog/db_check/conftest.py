from datetime import UTC, datetime

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.network_policy import NetworkPolicy
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutSampler

from tests.model_registry.constants import MR_OPERATOR_NAME
from utilities.constants import Labels
from utilities.general import wait_for_pods_by_labels

from .utils import extract_secret_values

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="class")
def model_catalog_postgres_secret(admin_client: DynamicClient, model_registry_namespace: str) -> Secret:
    """Get the model-catalog-postgres secret from model registry namespace"""
    return Secret(
        client=admin_client,
        name="model-catalog-postgres",
        namespace=model_registry_namespace,
        ensure_exists=True,
    )


@pytest.fixture(scope="class")
def model_catalog_postgres_secret_values(model_catalog_postgres_secret: Secret) -> dict[str, str]:
    """Capture current values of model-catalog-postgres secret in model registry namespace"""
    return extract_secret_values(secret=model_catalog_postgres_secret)


@pytest.fixture(scope="class")
def recreated_model_catalog_postgres_secret(
    admin_client: DynamicClient, model_catalog_postgres_secret: Secret
) -> dict[str, str]:
    """Delete model-catalog-postgres secret and wait for it to be recreated"""
    model_registry_namespace = py_config["model_registry_namespace"]
    resource_name = "model-catalog-postgres"

    LOGGER.info(f"Deleting secret {resource_name} in namespace {model_registry_namespace}")
    model_catalog_postgres_secret.delete()

    # Wait for the secret to be recreated by the operator
    LOGGER.info(f"Waiting for secret {resource_name} to be recreated...")

    recreated_secret = None
    for secret in TimeoutSampler(
        wait_timeout=120,
        sleep=10,
        func=Secret,
        client=admin_client,
        name=resource_name,
        namespace=model_registry_namespace,
    ):
        if secret.exists:
            LOGGER.info(f"Secret {resource_name} has been recreated")
            recreated_secret = secret
            break

    return extract_secret_values(secret=recreated_secret)


@pytest.fixture()
def model_catalog_network_policy(
    request: pytest.FixtureRequest, admin_client: DynamicClient, model_registry_namespace: str
) -> NetworkPolicy:
    """Get a model-catalog NetworkPolicy by name (parameterized)"""
    return NetworkPolicy(
        client=admin_client,
        name=request.param,
        namespace=model_registry_namespace,
        ensure_exists=True,
    )


@pytest.fixture(scope="class")
def deleted_network_policy_original_spec(
    request: pytest.FixtureRequest, admin_client: DynamicClient, model_registry_namespace: str
) -> dict:
    """Save the NetworkPolicy spec and owner references, then delete it. Returns the originals."""
    np = NetworkPolicy(
        client=admin_client,
        name=request.param,
        namespace=model_registry_namespace,
        ensure_exists=True,
    )
    original = {
        "name": request.param,
        "spec": np.instance.spec.to_dict(),
        "ownerReferences": [ref.to_dict() for ref in (np.instance.metadata.ownerReferences or [])],
    }

    LOGGER.info(f"Deleting NetworkPolicy {request.param}")
    original["deleted_at"] = datetime.now(tz=UTC)
    np.delete()

    return original


@pytest.fixture(scope="class")
def recreated_network_policy(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    deleted_network_policy_original_spec,
    model_registry_namespace: str,
) -> NetworkPolicy:
    """Wait for a deleted NetworkPolicy to be recreated and return it."""
    for np in TimeoutSampler(
        wait_timeout=15,
        sleep=5,
        func=NetworkPolicy,
        client=admin_client,
        name=request.param,
        namespace=model_registry_namespace,
    ):
        if np.exists:
            LOGGER.info(f"NetworkPolicy {request.param} has been recreated by operator")
            return np


@pytest.fixture()
def recreated_network_policy_scope_function(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    deleted_network_policy_original_spec,
    model_registry_namespace: str,
) -> NetworkPolicy:
    """Wait for a deleted NetworkPolicy to be recreated and return it (function-scoped)."""
    for np in TimeoutSampler(
        wait_timeout=60,
        sleep=5,
        func=NetworkPolicy,
        client=admin_client,
        name=request.param,
        namespace=model_registry_namespace,
    ):
        if np.exists:
            LOGGER.info(f"NetworkPolicy {request.param} has been recreated by operator")
            return np


@pytest.fixture(scope="class")
def restarted_operator_pod(admin_client: DynamicClient) -> Pod:
    """Restart the model registry operator pod and wait for it to be running."""
    operator_pods = wait_for_pods_by_labels(
        admin_client=admin_client,
        namespace=py_config["applications_namespace"],
        label_selector=f"{Labels.OpenDataHubIo.NAME}={MR_OPERATOR_NAME}",
        expected_num_pods=1,
    )
    operator_pods[0].delete()
    new_pod = wait_for_pods_by_labels(
        admin_client=admin_client,
        namespace=py_config["applications_namespace"],
        label_selector=f"{Labels.OpenDataHubIo.NAME}={MR_OPERATOR_NAME}",
        expected_num_pods=1,
    )[0]
    new_pod.wait_for_status(status=Pod.Status.RUNNING)
    return new_pod


@pytest.fixture()
def non_catalog_network_policy(admin_client: DynamicClient, model_registry_namespace: str) -> NetworkPolicy:
    """Create a NetworkPolicy without catalog labels in the model registry namespace."""
    with NetworkPolicy(
        client=admin_client,
        name="non-catalog-test-np",
        namespace=model_registry_namespace,
        pod_selector={"matchLabels": {"app": "non-catalog-app"}},
    ) as np:
        yield np
