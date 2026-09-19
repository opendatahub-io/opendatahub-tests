"""Health reporting, connection pooling and durability of the SQL backends.

These are the properties the shared SQLAlchemy Core base added on top of the
per-backend code it replaced: a pooled engine sized from the environment, a
storage health check that reflects the selected backend, and data that lives in
the database rather than in the pod.
"""

from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod

from tests.ai_safety.trustyai_service.storage_backends.constants import (
    ENV_MAX_OVERFLOW,
    ENV_POOL_SIZE,
    HEALTH_STATUS_OK,
    POSTGRES_SERVICE_DEPLOYMENT,
    STORAGE_CHECK_NAME,
)
from tests.ai_safety.trustyai_service.storage_backends.utils import (
    StorageBackendClient,
    build_upload_payload,
    wait_for_deployment_pods_ready,
    wait_for_model_observations,
)

LOGGER = structlog.get_logger(name=__name__)

# Deliberately smaller than the number of concurrent writers below, so the writers
# have to queue on the pool instead of each getting their own connection.
POOL_SIZE: str = "2"
MAX_OVERFLOW: str = "1"
CONCURRENT_WRITERS: int = 8


@pytest.mark.ai_safety
@pytest.mark.tier1
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-trustyai-storage-health"})],
    indirect=True,
)
class TestStorageHealthAndDurability:
    """Health reporting and the persistence guarantee that motivates a real database."""

    def test_readiness_reports_storage_backend(self, postgres_storage_client: StorageBackendClient) -> None:
        """Readiness includes a storage check, and it is green against a reachable database."""
        response = postgres_storage_client.readiness()
        assert response.status_code == HTTPStatus.OK, f"Readiness probe failed: {response.text}"

        checks = {check["name"]: check["status"] for check in response.json()["checks"]}
        assert STORAGE_CHECK_NAME in checks, f"No storage check in the readiness payload: {response.json()}"
        assert checks[STORAGE_CHECK_NAME] == HEALTH_STATUS_OK

    def test_general_health_is_healthy(self, postgres_storage_client: StorageBackendClient) -> None:
        """The combined health endpoint aggregates readiness and liveness."""
        response = postgres_storage_client.health()
        assert response.status_code == HTTPStatus.OK, f"Health endpoint failed: {response.text}"

        body = response.json()
        assert body["status"] == "healthy", f"Service reports unhealthy: {body}"
        assert body["checks"]["readiness"], "Readiness checks missing from the health payload"
        assert body["checks"]["liveness"], "Liveness checks missing from the health payload"

    def test_data_survives_service_restart(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        postgres_backed_service: Deployment,
        postgres_storage_client: StorageBackendClient,
    ) -> None:
        """Rows live in PostgreSQL, not in the pod: a fresh pod still sees them."""
        model_name = "postgres-restart-model"
        n_rows = 4

        assert (
            postgres_storage_client.upload(
                payload=build_upload_payload(model_name=model_name, n_rows=n_rows)
            ).status_code
            == HTTPStatus.OK
        )
        wait_for_model_observations(storage_client=postgres_storage_client, model_name=model_name, expected=n_rows)

        for pod in Pod.get(
            client=admin_client,
            namespace=model_namespace.name,
            label_selector=f"name={POSTGRES_SERVICE_DEPLOYMENT}",
        ):
            LOGGER.info(f"Deleting service pod {pod.name} to force a restart")
            pod.delete(wait=True)

        postgres_backed_service.wait_for_replicas(deployed=True, timeout=600)
        wait_for_deployment_pods_ready(
            client=admin_client,
            namespace=model_namespace.name,
            label_selector=f"name={POSTGRES_SERVICE_DEPLOYMENT}",
        )

        model_info = wait_for_model_observations(
            storage_client=postgres_storage_client, model_name=model_name, expected=n_rows
        )
        assert model_info["data"]["observations"] == n_rows, (
            "Data uploaded before the restart is missing after it, so it was not persisted to PostgreSQL"
        )


@pytest.mark.ai_safety
@pytest.mark.tier2
@pytest.mark.parametrize(
    "model_namespace, postgres_backed_service",
    [
        pytest.param(
            {"name": "test-trustyai-storage-pooling"},
            {"env": {ENV_POOL_SIZE: POOL_SIZE, ENV_MAX_OVERFLOW: MAX_OVERFLOW}},
        )
    ],
    indirect=True,
)
class TestStorageConnectionPooling:
    """Writers share a deliberately small pool instead of opening a connection each."""

    def test_pool_env_is_applied(self, postgres_backed_service: Deployment) -> None:
        """The pool sizing env reaches the container that builds the engine."""
        container = postgres_backed_service.instance.spec.template.spec.containers[0]
        env = {var["name"]: var.get("value") for var in container.env}
        assert env.get(ENV_POOL_SIZE) == POOL_SIZE
        assert env.get(ENV_MAX_OVERFLOW) == MAX_OVERFLOW

    @pytest.mark.dependency(name="concurrent_uploads_share_the_pool", scope="class")
    def test_concurrent_uploads_share_the_pool(self, postgres_storage_client: StorageBackendClient) -> None:
        """More concurrent writers than connections still all succeed."""
        model_name = "postgres-pool-model"
        rows_per_writer = 2

        def _upload(offset: int) -> Any:
            return postgres_storage_client.upload(
                payload=build_upload_payload(
                    model_name=model_name,
                    n_rows=rows_per_writer,
                    offset=offset * rows_per_writer,
                )
            )

        with ThreadPoolExecutor(max_workers=CONCURRENT_WRITERS) as executor:
            responses = list(executor.map(_upload, range(CONCURRENT_WRITERS)))

        failures = [response for response in responses if response.status_code != HTTPStatus.OK]
        assert not failures, f"{len(failures)} concurrent uploads failed, first: {failures[0].text}"

        expected = CONCURRENT_WRITERS * rows_per_writer
        model_info = wait_for_model_observations(
            storage_client=postgres_storage_client, model_name=model_name, expected=expected, timeout=180
        )
        assert model_info["data"]["observations"] == expected, (
            f"Expected {expected} rows from {CONCURRENT_WRITERS} concurrent writers"
        )

    @pytest.mark.dependency(depends=["concurrent_uploads_share_the_pool"], scope="class")
    def test_storage_stays_healthy_after_concurrent_load(self, postgres_storage_client: StorageBackendClient) -> None:
        """The pool is returned to a usable state once the writers finish."""
        response = postgres_storage_client.readiness()
        assert response.status_code == HTTPStatus.OK, f"Readiness failed after concurrent load: {response.text}"
        checks = {check["name"]: check["status"] for check in response.json()["checks"]}
        assert checks.get(STORAGE_CHECK_NAME) == HEALTH_STATUS_OK, (
            f"Storage unhealthy after concurrent load, pool may be exhausted: {response.json()}"
        )
