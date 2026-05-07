import math
import re
from datetime import UTC, datetime

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.network_policy import NetworkPolicy
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_registry.model_catalog.constants import CATALOG_CONTAINER
from tests.model_registry.model_catalog.db_check.utils import (
    find_language_mismatches_between_api_and_db,
    parse_language_properties_from_db,
)
from tests.model_registry.model_catalog.db_constants import LANGUAGE_PROPERTIES_DB_QUERY
from tests.model_registry.model_catalog.utils import execute_database_query, get_postgres_pod_in_namespace
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


class TestModelCatalogLoaderHealth:
    def test_no_duplicate_key_errors_in_catalog_logs(self, model_catalog_pod):
        """Given a model catalog pod with default data loaded
        When checking the catalog container logs
        Then no 'duplicated key not allowed' errors should be present
        """
        catalog_log = model_catalog_pod.log(container=CATALOG_CONTAINER)
        assert "duplicated key not allowed" not in catalog_log, (
            "Found duplicate key errors in catalog logs — property upsert may have regressed"
        )


@pytest.mark.parametrize(
    "model_catalog_network_policy",
    [
        pytest.param("model-catalog-postgres", id="test_network_policy_postgres"),
        pytest.param("model-catalog-https-route", id="test_network_policy_https_route"),
    ],
    indirect=True,
)
class TestModelCatalogDBNetworkPolicy:
    def test_network_policy_exists(self, model_catalog_network_policy):
        """Test that model-catalog NetworkPolicy exists and is accessible"""
        assert model_catalog_network_policy.exists, f"{model_catalog_network_policy.name} NetworkPolicy should exist"

    @pytest.mark.test_postgres_network_policy_only
    def test_postgres_network_policy_restricts_to_port_5432(self, model_catalog_network_policy):
        """Test that NetworkPolicy only allows TCP 5432 ingress"""
        spec = model_catalog_network_policy.instance.spec
        assert "Ingress" in spec.policyTypes, "NetworkPolicy should have Ingress policy type"
        assert len(spec.ingress) == 1, "NetworkPolicy should have exactly one ingress rule"

        port = spec.ingress[0].ports[0]
        assert port.port == 5432, "NetworkPolicy should allow only PostgreSQL port 5432"
        assert port.protocol == "TCP", "NetworkPolicy port should use TCP protocol"

    @pytest.mark.test_postgres_network_policy_only
    def test_network_policy_allows_only_catalog_pods(self, model_catalog_network_policy):
        """Test that only model-catalog pods can reach postgres"""
        from_selector = model_catalog_network_policy.instance.spec.ingress[0]["from"][0].podSelector.matchLabels
        assert from_selector["component"] == "model-catalog", (
            "Only model-catalog pods should be allowed to access postgres"
        )

    @pytest.mark.test_https_route_network_policy_only
    def test_https_route_ingress_namespace_labels(self, model_catalog_network_policy):
        """Given the model-catalog-https-route NetworkPolicy
        When inspecting its ingress namespace selectors
        Then it allows traffic from dashboard namespace and OpenShift ingress namespace
        """
        from_rules = model_catalog_network_policy.instance.spec.ingress[0]["from"]
        namespace_labels = [
            rule.namespaceSelector.matchLabels
            for rule in from_rules
            if hasattr(rule, "namespaceSelector") and rule.namespaceSelector
        ]
        assert any(labels.get("opendatahub.io/generated-namespace") == "true" for labels in namespace_labels), (
            "NetworkPolicy should allow traffic from dashboard namespace (opendatahub.io/generated-namespace: true)"
        )
        assert any(labels.get("network.openshift.io/policy-group") == "ingress" for labels in namespace_labels), (
            "NetworkPolicy should allow traffic from OpenShift ingress namespace"
        )


class TestModelCatalogNetworkPolicyConnectivity:
    def test_dashboard_can_reach_model_catalog(self, dashboard_pod, model_registry_namespace: str):
        """Given a dashboard pod in the applications namespace
        When curling the model-catalog internal service on the kube-rbac-proxy port
        Then the connection is not blocked by the NetworkPolicy
        """
        service_url = f"https://model-catalog.{model_registry_namespace}.svc.cluster.local:8443"
        result = dashboard_pod.execute(command=["curl", "-k", "--connect-timeout", "10", service_url])
        LOGGER.info(f"curl to {service_url}: rc={result.rc}, stdout={result.stdout}, stderr={result.stderr}")
        assert result.rc == 0, (
            f"Dashboard pod cannot reach model-catalog at {service_url} — "
            f"NetworkPolicy may be blocking traffic (rc={result.rc}, stderr={result.stderr})"
        )
        assert "Connection timed out" not in result.stdout, (
            f"Dashboard pod connection timed out to model-catalog at {service_url} — "
            f"NetworkPolicy may be blocking traffic"
        )


@pytest.mark.parametrize(
    "deleted_network_policy_original_spec, recreated_network_policy",
    [
        pytest.param("model-catalog-postgres", "model-catalog-postgres", id="test_network_policy_postgres"),
        pytest.param("model-catalog-https-route", "model-catalog-https-route", id="test_network_policy_https_route"),
    ],
    indirect=True,
)
class TestModelCatalogDBNetworkPolicyRecreation:
    def test_network_policy_recreated_after_deletion(
        self,
        recreated_network_policy,
    ):
        """Test that operator recreates NetworkPolicy within ~10 seconds after deletion"""
        assert recreated_network_policy.exists, (
            f"{recreated_network_policy.name} NetworkPolicy should have been recreated"
        )

    def test_network_policy_spec_intact_after_recreation(
        self,
        deleted_network_policy_original_spec,
        recreated_network_policy,
    ):
        """Test that recreated NetworkPolicy has identical spec to original"""
        recreated_spec = recreated_network_policy.instance.spec.to_dict()
        assert recreated_spec == deleted_network_policy_original_spec["spec"], (
            f"Recreated {recreated_network_policy.name} spec does not match original"
        )

    def test_network_policy_owner_references_after_recreation(
        self,
        deleted_network_policy_original_spec,
        recreated_network_policy,
    ):
        """Test that recreated NetworkPolicy maintains correct owner references"""
        owner_refs = [ref.to_dict() for ref in (recreated_network_policy.instance.metadata.ownerReferences or [])]
        assert owner_refs, f"Recreated {recreated_network_policy.name} should have owner references"
        assert owner_refs == deleted_network_policy_original_spec["ownerReferences"], (
            f"Recreated {recreated_network_policy.name} owner references do not match original"
        )


@pytest.mark.parametrize(
    "deleted_network_policy_original_spec, recreated_network_policy_scope_function",
    [
        pytest.param("model-catalog-postgres", "model-catalog-postgres", id="test_network_policy_postgres"),
        pytest.param("model-catalog-https-route", "model-catalog-https-route", id="test_network_policy_https_route"),
    ],
    indirect=True,
)
class TestModelCatalogDBNetworkPolicyNoReconciliationStorm:
    def test_no_reconciliation_storm_after_network_policy_recreation(
        self,
        restarted_operator_pod,
        deleted_network_policy_original_spec,
        recreated_network_policy_scope_function,
        model_registry_operator_pod,
    ):
        """Test that no reconciliation storm occurs after NetworkPolicy deletion and recreation"""
        timestamp_before = deleted_network_policy_original_spec["deleted_at"]
        since_seconds = math.ceil((datetime.now(tz=UTC) - timestamp_before).total_seconds()) + 5
        logs = model_registry_operator_pod.log(container="manager", since_seconds=since_seconds)
        LOGGER.info(f"Operator logs (last {since_seconds}s):\n{logs}")

        np_name = recreated_network_policy_scope_function.name
        np_creation_lines = [
            line
            for line in logs.splitlines()
            if "Kind=NetworkPolicy" in line and np_name in line and "creating" in line.lower()
        ]
        assert len(np_creation_lines) == 1, (
            f"Expected exactly 1 creation event for {np_name}, got {len(np_creation_lines)}: {np_creation_lines}"
        )

        reconcile_id = re.search(r'"reconcileID":\s*"([^"]+)"', np_creation_lines[0]).group(1)
        reconcile_log_lines = [line for line in logs.splitlines() if reconcile_id in line]
        LOGGER.info(f"ReconcileID {reconcile_id} has {len(reconcile_log_lines)} log entries")
        assert len(reconcile_log_lines) == 4, (
            f"Expected 4 log entries for reconcileID {reconcile_id}, "
            f"got {len(reconcile_log_lines)}: {reconcile_log_lines}"
        )


class TestNonCatalogNetworkPolicyNotRecreated:
    def test_non_catalog_network_policy_not_recreated(
        self,
        admin_client: DynamicClient,
        non_catalog_network_policy,
        model_registry_namespace: str,
    ):
        """Test that operator does not recreate a NetworkPolicy without catalog labels"""
        non_catalog_network_policy.delete()
        try:
            for sample in TimeoutSampler(
                wait_timeout=30,
                sleep=5,
                func=NetworkPolicy,
                client=admin_client,
                name=non_catalog_network_policy.name,
                namespace=model_registry_namespace,
            ):
                if sample.exists:
                    pytest.fail("Non-catalog NetworkPolicy should not be recreated by operator")
        except TimeoutExpiredError:
            LOGGER.info("Non-catalog NetworkPolicy was not recreated after 30 seconds, as expected")
        else:
            pytest.fail("Expected TimeoutExpiredError but sampler completed without it")


class TestLanguagePropertyConsistency:
    def test_language_properties_match_between_api_and_database(
        self,
        admin_client: DynamicClient,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        model_registry_namespace: str,
    ):
        """Given models loaded into the catalog
        When comparing language values from the API and database
        Then all models with language properties should have matching values
        """
        db_result = execute_database_query(
            admin_client=admin_client,
            query=LANGUAGE_PROPERTIES_DB_QUERY,
            namespace=model_registry_namespace,
        )
        db_languages = parse_language_properties_from_db(psql_output=db_result)
        assert db_languages, "No language properties found in database"
        LOGGER.info(f"Found language properties for {len(db_languages)} models in database")

        mismatches = find_language_mismatches_between_api_and_db(
            db_languages=db_languages,
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
        )
        assert not mismatches, (
            f"Language property mismatches between API and DB for {len(mismatches)} model(s):\n" + "\n".join(mismatches)
        )
