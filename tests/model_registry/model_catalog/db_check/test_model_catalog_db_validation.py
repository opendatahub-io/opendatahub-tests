import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.network_policy import NetworkPolicy
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler

from tests.model_registry.model_catalog.utils import get_postgres_pod_in_namespace
from tests.model_registry.utils import (
    wait_for_model_catalog_pod_ready_after_deletion,
)

LOGGER = get_logger(name=__name__)


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

    def test_postgres_network_policy_has_correct_labels(self, model_catalog_postgres_network_policy):
        """Test that NetworkPolicy has correct operator-managed labels"""
        labels = model_catalog_postgres_network_policy.instance.metadata.labels
        assert labels["app.kubernetes.io/created-by"] == "model-registry-operator", (
            "NetworkPolicy should be created by model-registry-operator"
        )
        assert labels["app.kubernetes.io/part-of"] == "model-catalog", "NetworkPolicy should be part of model-catalog"
        assert labels["app.kubernetes.io/managed-by"] == "model-registry-operator", (
            "NetworkPolicy should be managed by model-registry-operator"
        )

    @pytest.mark.dependency(name="test_postgres_network_policy_recreation")
    def test_postgres_network_policy_recreated_on_reconciliation(
        self,
        admin_client: DynamicClient,
        model_catalog_postgres_network_policy,
        model_registry_namespace: str,
    ):
        """Test that operator recreates NetworkPolicy when reconciliation is triggered.

        The NetworkPolicy is NOT watched directly by the operator, so deleting it alone
        won't trigger recreation. Deleting the postgres pod triggers a Deployment change,
        which triggers reconciliation and recreates the NetworkPolicy.
        """
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

    @pytest.mark.dependency(depends=["test_postgres_network_policy_recreation"])
    def test_postgres_network_policy_spec_preserved_after_recreation(
        self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
    ):
        """Test that recreated NetworkPolicy has the same correct spec"""
        recreated_np = NetworkPolicy(
            client=admin_client,
            name="model-catalog-postgres",
            namespace=model_registry_namespace,
        )
        assert recreated_np.exists, "Recreated NetworkPolicy should exist"

        spec = recreated_np.instance.spec

        # Verify podSelector targets postgres pods
        assert spec.podSelector.matchLabels["app.kubernetes.io/name"] == "model-catalog-postgres", (
            "Recreated NetworkPolicy should still target postgres pods"
        )

        # Verify ingress policy type
        assert "Ingress" in spec.policyTypes, "Recreated NetworkPolicy should have Ingress policy type"

        # Verify port restriction
        assert len(spec.ingress) == 1, "Recreated NetworkPolicy should have exactly one ingress rule"
        port = spec.ingress[0].ports[0]
        assert port.port == 5432, "Recreated NetworkPolicy should allow only PostgreSQL port 5432"
        assert port.protocol == "TCP", "Recreated NetworkPolicy port should use TCP protocol"

        # Verify from selector allows only catalog pods
        from_selector = spec.ingress[0]["from"][0].podSelector.matchLabels
        assert from_selector["component"] == "model-catalog", (
            "Recreated NetworkPolicy should still allow only model-catalog pods"
        )

        # Verify labels
        labels = recreated_np.instance.metadata.labels
        assert labels["app.kubernetes.io/created-by"] == "model-registry-operator"
        assert labels["app.kubernetes.io/part-of"] == "model-catalog"

        LOGGER.info("Recreated NetworkPolicy spec and labels match expected configuration")

    def test_postgres_network_policy_recreated_after_operator_restart(
        self,
        admin_client: DynamicClient,
        model_registry_operator_pod: Pod,
        model_registry_namespace: str,
    ):
        """Test that operator restart recreates a deleted NetworkPolicy via initial reconciliation.

        This simulates a production scenario where the operator pod is restarted
        (e.g., during upgrades) and must reconcile all managed resources including
        the NetworkPolicy.
        """
        # Delete the NetworkPolicy first
        np = NetworkPolicy(
            client=admin_client,
            name="model-catalog-postgres",
            namespace=model_registry_namespace,
        )
        assert np.exists, "NetworkPolicy should exist before operator restart"
        np.delete()

        # Restart the operator pod to trigger initial reconciliation
        LOGGER.info(f"Deleting operator pod {model_registry_operator_pod.name} to trigger reconciliation")
        model_registry_operator_pod.delete(wait=True)

        # Wait for the NetworkPolicy to be recreated
        for recreated_np in TimeoutSampler(
            wait_timeout=180,
            sleep=10,
            func=NetworkPolicy,
            client=admin_client,
            name="model-catalog-postgres",
            namespace=model_registry_namespace,
        ):
            if recreated_np.exists:
                LOGGER.info("NetworkPolicy has been recreated after operator restart")
                break
