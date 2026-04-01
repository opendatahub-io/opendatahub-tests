import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.network_policy import NetworkPolicy
from timeout_sampler import TimeoutSampler

from tests.model_registry.model_catalog.utils import get_postgres_pod_in_namespace
from tests.model_registry.utils import (
    wait_for_model_catalog_pod_ready_after_deletion,
)

LOGGER = structlog.get_logger(name=__name__)


class TestModelCatalogDBSecret:
    def test_model_catalog_postgres_secret_exists(self, model_catalog_postgres_secret_values):
        """Test that model-catalog-postgres secret exists and is accessible"""
        assert model_catalog_postgres_secret_values, (
            f"model-catalog-postgres secret should exist and be accessible: {model_catalog_postgres_secret_values}"
        )

    @pytest.mark.dependency(name="test_model_catalog_postgres_password_recreation")
    def test_model_catalog_postgres_password_recreation(
        self, model_catalog_postgres_secret_values, recreated_model_catalog_postgres_secret
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
    def test_model_catalog_pod_ready_after_secret_recreation(
        self, admin_client: DynamicClient, model_registry_namespace: str
    ):
        """Test that model catalog pod becomes ready after secret recreation"""
        # delete the postgres pod first
        get_postgres_pod_in_namespace(admin_client=admin_client, namespace=model_registry_namespace).delete()
        # Wait for model catalog pod to be ready after the secret deletion/recreation
        wait_for_model_catalog_pod_ready_after_deletion(
            client=admin_client, model_registry_namespace=model_registry_namespace
        )
        LOGGER.info("Model catalog pod is ready after secret recreation")


class TestModelCatalogDBNetworkPolicy:
    def test_postgres_network_policy_exists(self, model_catalog_postgres_network_policy):
        """Test that postgres NetworkPolicy exists and is accessible"""
        assert model_catalog_postgres_network_policy.exists, "model-catalog-postgres NetworkPolicy should exist"

    def test_postgres_network_policy_restricts_to_port_5432(self, model_catalog_postgres_network_policy):
        """Test that NetworkPolicy only allows TCP 5432 ingress"""
        spec = model_catalog_postgres_network_policy.instance.spec
        assert "Ingress" in spec.policyTypes, "NetworkPolicy should have Ingress policy type"
        assert len(spec.ingress) == 1, "NetworkPolicy should have exactly one ingress rule"

        port = spec.ingress[0].ports[0]
        assert port.port == 5432, "NetworkPolicy should allow only PostgreSQL port 5432"
        assert port.protocol == "TCP", "NetworkPolicy port should use TCP protocol"

    def test_postgres_network_policy_allows_only_catalog_pods(self, model_catalog_postgres_network_policy):
        """Test that only model-catalog pods can reach postgres"""
        from_selector = model_catalog_postgres_network_policy.instance.spec.ingress[0]["from"][
            0
        ].podSelector.matchLabels
        assert from_selector["component"] == "model-catalog", (
            "Only model-catalog pods should be allowed to access postgres"
        )

    @pytest.mark.dependency(name="test_postgres_network_policy_recreation")
    def test_postgres_network_policy_recreated_after_deletion(
        self,
        admin_client: DynamicClient,
        model_catalog_postgres_network_policy,
        model_registry_namespace: str,
    ):
        """Test that operator recreates NetworkPolicy after deletion"""
        model_catalog_postgres_network_policy.delete()
        get_postgres_pod_in_namespace(admin_client=admin_client, namespace=model_registry_namespace).delete()
        for np in TimeoutSampler(
            wait_timeout=120,
            sleep=10,
            func=NetworkPolicy,
            client=admin_client,
            name="model-catalog-postgres",
            namespace=model_registry_namespace,
        ):
            if np.exists:
                LOGGER.info("NetworkPolicy has been recreated by operator")
                break
