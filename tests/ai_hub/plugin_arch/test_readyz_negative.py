from typing import Self

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod
from timeout_sampler import TimeoutSampler

from tests.ai_hub.plugin_arch.utils import (
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

    def test_readyz_reports_unhealthy_during_db_outage(
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
        assert body.get("status") == "not_ready", f"/readyz returned 503 but status is '{body.get('status')}'"
        LOGGER.info(f"/readyz returned 503 with body: {body}")

        healthz_response = requests.get(healthz_url, headers=model_registry_rest_headers, verify=False, timeout=10)
        assert healthz_response.ok, (
            f"/healthz should remain healthy during DB outage, got {healthz_response.status_code}"
        )


class TestReadyzAfterPodRestart:
    """Tests for /readyz behavior after catalog pod restart with DB outage (RHOAIENG-67494)."""

    def test_readyz_unhealthy_after_pod_restart_and_db_revoke(
        self: Self,
        admin_client: DynamicClient,
        catalog_base_url: str,
        model_registry_rest_headers: dict[str, str],
        model_registry_namespace: str,
        healthy_catalog_state: None,
    ) -> None:
        """
        Given a healthy catalog pod with /readyz returning 200
        When the pod is deleted and a new one starts
        And the database login is revoked after the new pod begins listening
        Then /readyz returns 503 on the new pod
        """
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

        new_pod = get_model_catalog_pod(client=admin_client, model_registry_namespace=model_registry_namespace)[0]
        for readyz_sample in TimeoutSampler(
            wait_timeout=READYZ_RECOVERY_TIMEOUT,
            sleep=5,
            func=new_pod.execute,
            command=[
                "curl",
                "-s",
                "--connect-timeout",
                "5",
                "--max-time",
                "10",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                "http://localhost:8080/readyz",
            ],
            container="catalog",
            ignore_rc=True,
        ):
            if readyz_sample.strip() == "200":
                LOGGER.info("/readyz on new pod returned 200, server is listening")
                break

        LOGGER.info("Revoking catalog_user login after new pod is serving")
        run_superuser_sql(admin_client=admin_client, namespace=model_registry_namespace, sql=REVOKE_LOGIN_SQL)

        for readyz_sample in TimeoutSampler(
            wait_timeout=READYZ_UNHEALTHY_TIMEOUT,
            sleep=5,
            func=new_pod.execute,
            command=[
                "curl",
                "-s",
                "--connect-timeout",
                "5",
                "--max-time",
                "10",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                "http://localhost:8080/readyz",
            ],
            container="catalog",
            ignore_rc=True,
        ):
            if readyz_sample.strip() == "503":
                LOGGER.info("/readyz on new pod returned 503 after DB revoke")
                break
