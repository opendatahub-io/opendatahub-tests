import pytest
from kubernetes.dynamic import DynamicClient
from simple_logger.logger import get_logger

from tests.model_registry.model_catalog.utils import get_postgres_pod_in_namespace
from tests.model_registry.utils import (
    wait_for_model_catalog_pod_ready_after_deletion,
)

LOGGER = get_logger(name=__name__)


def test_model_catalog_postgres_secret_exists(model_catalog_postgres_secret_values):
    """Test that model-catalog-postgres secret exists and is accessible"""
    secret_values = model_catalog_postgres_secret_values
    assert secret_values, "model-catalog-postgres secret should exist and be accessible"

    # Log the keys (not values) for debugging
    LOGGER.info(f"Secret contains keys: {list(secret_values.keys())}")


@pytest.mark.dependency(name="test_model_catalog_postgres_password_recreation")
def test_model_catalog_postgres_password_recreation(
    model_catalog_postgres_secret_values, recreated_model_catalog_postgres_secret
):
    """Test that secret recreation generates new password but preserves user/database name"""
    # Verify database-name and database-user did NOT change
    unchanged_keys = ["database-name", "database-user"]
    for key in unchanged_keys:
        assert model_catalog_postgres_secret_values[key] == recreated_model_catalog_postgres_secret[key], (
            f"{key} should remain the same after secret recreation"
        )

    # Verify database-password DID change (randomization working)
    assert (
        model_catalog_postgres_secret_values["database-password"]
        != recreated_model_catalog_postgres_secret["database-password"]
    ), "database-password should be different after secret recreation (randomized)"

    LOGGER.info("Password randomization verified - new password generated on recreation")


@pytest.mark.dependency(depends=["test_model_catalog_postgres_password_recreation"])
def test_model_catalog_pod_ready_after_secret_recreation(admin_client: DynamicClient, model_registry_namespace: str):
    """Test that model catalog pod becomes ready after secret recreation"""
    # delete the postgres pod first
    get_postgres_pod_in_namespace(admin_client=admin_client).delete()
    # Wait for model catalog pod to be ready after the secret deletion/recreation
    wait_for_model_catalog_pod_ready_after_deletion(
        client=admin_client, model_registry_namespace=model_registry_namespace
    )
    LOGGER.info("Model catalog pod is ready after secret recreation")
