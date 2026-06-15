from typing import Self

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod
from timeout_sampler import TimeoutSampler

from tests.ai_hub.model_catalog.plugin_arch.utils import (
    READYZ_RECOVERY_TIMEOUT,
    poll_readyz,
    run_superuser_sql,
)
from tests.ai_hub.utils import get_model_catalog_pod

LOGGER = structlog.get_logger(name=__name__)

READYZ_UNHEALTHY_TIMEOUT: int = 120

REVOKE_LOGIN_SQL: str = (
    "ALTER USER catalog_user NOLOGIN;"
    " SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE usename = 'catalog_user';"
)


pytestmark = [
    pytest.mark.tier3,
    pytest.mark.skip_must_gather,
    pytest.mark.usefixtures(
        "updated_dsc_component_state_scope_session",
        "model_registry_namespace",
    ),
]


class TestReadyzDuringDatabaseOutage:
    """Negative tests for /readyz behavior when the database is unavailable (RHOAIENG-67494)."""

    def test_readyz_reports_unhealthy_during_db_outage_and_recovers(
        self: Self,
        admin_client: DynamicClient,
        catalog_base_url: str,
        model_registry_rest_headers: dict[str, str],
        model_registry_namespace: str,
        healthy_catalog_state: None,
    ) -> None:
        """
        Given a healthy catalog server with /readyz returning 200
        When the database user login is revoked and connections terminated
        Then /readyz returns 503 with status 'not_ready'
        And /healthz continues to return 200
        """
        readyz_url = f"{catalog_base_url}/readyz"
        healthz_url = f"{catalog_base_url}/healthz"

        LOGGER.info("Revoking catalog_user login and terminating connections")
        run_superuser_sql(admin_client=admin_client, namespace=model_registry_namespace, sql=REVOKE_LOGIN_SQL)

        response = poll_readyz(
            url=readyz_url,
            headers=model_registry_rest_headers,
            expected_code=503,
            timeout=READYZ_UNHEALTHY_TIMEOUT,
        )
        body = response.json()
        assert body["status"] == "not_ready", f"/readyz returned 503 but status is '{body['status']}'"
        LOGGER.info(f"/readyz returned 503 with body: {body}")

        healthz_response = requests.get(healthz_url, headers=model_registry_rest_headers, verify=False, timeout=10)
        assert healthz_response.ok, (
            f"/healthz should remain healthy during DB outage, got {healthz_response.status_code}"
        )


class TestReadyzColdStart:
    """Tests for /readyz behavior during catalog pod cold start (RHOAIENG-67494)."""

    def test_readyz_starts_unhealthy_and_recovers(
        self: Self,
        admin_client: DynamicClient,
        catalog_base_url: str,
        model_registry_rest_headers: dict[str, str],
        model_registry_namespace: str,
        healthy_catalog_state: None,
    ) -> None:
        """
        Given the database login is revoked
        When the catalog pod is deleted and a new one starts
        Then /readyz returns 503 on the new pod (starts unhealthy, DB unreachable)
        """
        LOGGER.info("Revoking catalog_user login before pod restart")
        run_superuser_sql(admin_client=admin_client, namespace=model_registry_namespace, sql=REVOKE_LOGIN_SQL)

        catalog_pods = get_model_catalog_pod(client=admin_client, model_registry_namespace=model_registry_namespace)
        assert catalog_pods, "No catalog pods found"
        original_pod_name = catalog_pods[0].name

        LOGGER.info(f"Deleting catalog pod '{original_pod_name}'")
        catalog_pods[0].delete()

        for sample in TimeoutSampler(
            wait_timeout=READYZ_RECOVERY_TIMEOUT,
            sleep=5,
            func=get_model_catalog_pod,
            client=admin_client,
            model_registry_namespace=model_registry_namespace,
        ):
            if sample and sample[0].name != original_pod_name and sample[0].status == Pod.Status.RUNNING:
                LOGGER.info(f"New catalog pod running: {sample[0].name}")
                break

        # Exec directly on the pod because the route is unreachable — the readiness probe
        # fails (503), so OpenShift removes the pod from service endpoints.
        new_pod = get_model_catalog_pod(client=admin_client, model_registry_namespace=model_registry_namespace)[0]
        readyz_output = new_pod.execute(
            command=["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "http://localhost:8080/readyz"],
            container="catalog",
            ignore_rc=True,
        )
        LOGGER.info(f"/readyz on new pod returned HTTP {readyz_output}")
        assert readyz_output.strip() != "200", f"New pod should start unhealthy but /readyz returned {readyz_output}"


class TestReadyzHeartbeatLoss:
    """Tests for /readyz behavior when leader loses database connectivity (RHOAIENG-67494)."""

    def test_readyz_unhealthy_on_heartbeat_failure(
        self: Self,
        admin_client: DynamicClient,
        catalog_base_url: str,
        model_registry_rest_headers: dict[str, str],
        model_registry_namespace: str,
        healthy_catalog_state: None,
    ) -> None:
        """
        Given the catalog server is the active leader with /readyz returning 200
        When the database becomes unavailable (heartbeat cannot be sent)
        Then /readyz returns 503 after consecutive heartbeat failures
        """
        readyz_url = f"{catalog_base_url}/readyz"

        LOGGER.info("Revoking catalog_user login to cause heartbeat failure")
        run_superuser_sql(admin_client=admin_client, namespace=model_registry_namespace, sql=REVOKE_LOGIN_SQL)

        response = poll_readyz(
            url=readyz_url,
            headers=model_registry_rest_headers,
            expected_code=503,
            timeout=READYZ_UNHEALTHY_TIMEOUT,
        )
        LOGGER.info(f"/readyz returned 503 after heartbeat loss: {response.json()}")
