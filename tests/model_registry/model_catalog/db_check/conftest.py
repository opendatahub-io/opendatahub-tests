import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.network_policy import NetworkPolicy
from ocp_resources.secret import Secret
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutSampler

from utilities.opendatahub_logger import get_logger

from .utils import extract_secret_values

LOGGER = get_logger(name=__name__)


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


@pytest.fixture(scope="class")
def model_catalog_postgres_network_policy(admin_client: DynamicClient, model_registry_namespace: str) -> NetworkPolicy:
    """Get the model-catalog-postgres NetworkPolicy from model registry namespace"""
    return NetworkPolicy(
        client=admin_client,
        name="model-catalog-postgres",
        namespace=model_registry_namespace,
        ensure_exists=True,
    )
