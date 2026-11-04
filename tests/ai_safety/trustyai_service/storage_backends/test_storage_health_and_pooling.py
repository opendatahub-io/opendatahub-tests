"""Health, concurrency, and persistence for CR-managed PostgreSQL storage."""

from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod

from tests.ai_safety.trustyai_service.storage_backends.constants import HEALTH_STATUS_OK, STORAGE_CHECK_NAME
from tests.ai_safety.trustyai_service.storage_backends.utils import (
    StorageBackendClient,
    build_upload_payload,
    deployment_pod_selector,
    wait_for_model_observations,
)

CONCURRENT_WRITERS: int = 8


@pytest.mark.ai_safety
@pytest.mark.tier1
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-trustyai-storage-health"})],
    indirect=True,
)
class TestStorageHealthAndDurability:
    """Health and persistence guarantee for CR-managed PostgreSQL storage."""

    def test_readiness_reports_storage_backend(self, postgres_storage_client: StorageBackendClient) -> None:
        response = postgres_storage_client.readiness()
        assert response.status_code == HTTPStatus.OK, f"Readiness probe failed: {response.text}"

        checks = {check["name"]: check["status"] for check in response.json()["checks"]}
        assert checks.get(STORAGE_CHECK_NAME) == HEALTH_STATUS_OK, f"Storage is not ready: {response.json()}"

    def test_general_health_is_healthy(self, postgres_storage_client: StorageBackendClient) -> None:
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
        """Rows persist in the CR-configured database after the operator replaces the service pod."""
        model_name = "database-restart-model"
        n_rows = 4
        response = postgres_storage_client.upload(payload=build_upload_payload(model_name=model_name, n_rows=n_rows))
        assert response.status_code == HTTPStatus.OK, f"Upload failed: {response.text}"
        wait_for_model_observations(storage_client=postgres_storage_client, model_name=model_name, expected=n_rows)

        selector = deployment_pod_selector(deployment=postgres_backed_service)
        pods = list(Pod.get(client=admin_client, namespace=model_namespace.name, label_selector=selector))
        assert pods, f"No operator-managed TrustyAI pod found with selector {selector}"
        pods[0].delete(wait=True)

        postgres_backed_service.wait_for_replicas(deployed=True, timeout=600)
        model_info = wait_for_model_observations(
            storage_client=postgres_storage_client,
            model_name=model_name,
            expected=n_rows,
            timeout=180,
        )
        assert model_info["data"]["observations"] == n_rows, (
            "Data uploaded before the pod restart is missing from the CR-configured database"
        )


@pytest.mark.ai_safety
@pytest.mark.tier2
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-trustyai-storage-pooling"})],
    indirect=True,
)
class TestStorageConnectionPooling:
    """Concurrent writers continue to succeed through the service's shared database pool."""

    @pytest.mark.dependency(name="concurrent_uploads_share_the_pool", scope="class")
    def test_concurrent_uploads_share_the_pool(self, postgres_storage_client: StorageBackendClient) -> None:
        model_name = "database-pool-model"
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
            storage_client=postgres_storage_client,
            model_name=model_name,
            expected=expected,
            timeout=180,
        )
        assert model_info["data"]["observations"] == expected

    @pytest.mark.dependency(depends=["concurrent_uploads_share_the_pool"], scope="class")
    def test_storage_stays_healthy_after_concurrent_load(self, postgres_storage_client: StorageBackendClient) -> None:
        response = postgres_storage_client.readiness()
        assert response.status_code == HTTPStatus.OK, f"Readiness failed after concurrent load: {response.text}"
        checks = {check["name"]: check["status"] for check in response.json()["checks"]}
        assert checks.get(STORAGE_CHECK_NAME) == HEALTH_STATUS_OK, (
            f"Storage unhealthy after concurrent load: {response.json()}"
        )
