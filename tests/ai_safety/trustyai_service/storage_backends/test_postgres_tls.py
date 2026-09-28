"""TLS posture of the SQL storage backends.

The service refuses a database connection it cannot authenticate: without a CA
certificate libpq would fall back to `sslmode=prefer`, which permits an
unverified and possibly plaintext connection. `DATABASE_ALLOW_INSECURE_TLS` is
the single, explicit opt-out.
"""

from collections.abc import Callable
from http import HTTPStatus

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret

from tests.ai_safety.trustyai_service.storage_backends.constants import (
    ENV_ALLOW_INSECURE_TLS,
    ENV_STORAGE_FORMAT,
    ENV_TLS_CA_CERT,
    HEALTH_STATUS_OK,
    SERVICE_DB_CA_FILE,
    SERVICE_DB_CA_MOUNT_PATH,
    STORAGE_CHECK_NAME,
    STORAGE_FORMAT_POSTGRES,
    STORAGE_SERVICE_NAME,
)
from tests.ai_safety.trustyai_service.storage_backends.utils import (
    StorageBackendClient,
    build_upload_payload,
    wait_for_model_observations,
    wait_for_pod_log_message,
)

LOGGER = structlog.get_logger(name=__name__)

NO_CA_DEPLOYMENT: str = f"{STORAGE_SERVICE_NAME}-no-ca"
INSECURE_DEPLOYMENT: str = f"{STORAGE_SERVICE_NAME}-insecure"
MISSING_CA_PATH_DEPLOYMENT: str = f"{STORAGE_SERVICE_NAME}-missing-ca-path"

# Emitted by storage/__init__.py::_tls_error and logged by the storage health check.
TLS_ERROR_FRAGMENT: str = "requires authenticated TLS but no CA certificate was found"


@pytest.mark.ai_safety
@pytest.mark.tier3
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-trustyai-postgres-tls"})],
    indirect=True,
)
class TestPostgresStorageTls:
    """Authenticated TLS is required unless the deployment opts out explicitly."""

    def test_service_refuses_to_start_without_ca(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        postgres_db: Deployment,
        postgres_db_credentials_secret: Secret,
        deploy_storage_service: Callable[..., object],
    ) -> None:
        """With no CA mounted and no opt-out, storage stays unready and says why."""
        with deploy_storage_service(
            name=NO_CA_DEPLOYMENT,
            env={ENV_STORAGE_FORMAT: STORAGE_FORMAT_POSTGRES},
            db_credentials_secret_name=postgres_db_credentials_secret.name,
            wait_for_ready=False,
        ) as deployment:
            logs = wait_for_pod_log_message(
                client=admin_client,
                namespace=model_namespace.name,
                label_selector=f"name={NO_CA_DEPLOYMENT}",
                message=TLS_ERROR_FRAGMENT,
            )
            assert SERVICE_DB_CA_FILE in logs, (
                f"The refusal should name the CA path the service looked at ({SERVICE_DB_CA_FILE}); "
                f"logs: {logs[-2000:]}"
            )
            assert not deployment.instance.status.get("readyReplicas"), (
                "Service became ready despite having no way to authenticate the database TLS certificate"
            )

    def test_service_refuses_when_configured_ca_path_is_absent(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        postgres_db: Deployment,
        postgres_db_credentials_secret: Secret,
        postgres_db_ca_secret: Secret,
        deploy_storage_service: Callable[..., object],
    ) -> None:
        """A CA is mounted, but DATABASE_TLS_CA_CERT points somewhere else, so it is unusable."""
        absent_path = "/etc/tls/db/does-not-exist.crt"

        with deploy_storage_service(
            name=MISSING_CA_PATH_DEPLOYMENT,
            env={ENV_STORAGE_FORMAT: STORAGE_FORMAT_POSTGRES, ENV_TLS_CA_CERT: absent_path},
            db_credentials_secret_name=postgres_db_credentials_secret.name,
            ca_secret_name=postgres_db_ca_secret.name,
            ca_mount_path=SERVICE_DB_CA_MOUNT_PATH,
            wait_for_ready=False,
        ) as deployment:
            logs = wait_for_pod_log_message(
                client=admin_client,
                namespace=model_namespace.name,
                label_selector=f"name={MISSING_CA_PATH_DEPLOYMENT}",
                message=TLS_ERROR_FRAGMENT,
            )
            assert absent_path in logs, (
                f"The refusal should name the configured path ({absent_path}); logs: {logs[-2000:]}"
            )
            assert not deployment.instance.status.get("readyReplicas"), (
                "Service became ready with DATABASE_TLS_CA_CERT pointing at a file that does not exist"
            )

    def test_insecure_opt_in_allows_startup(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        postgres_db: Deployment,
        postgres_db_credentials_secret: Secret,
        deploy_storage_service: Callable[..., object],
    ) -> None:
        """DATABASE_ALLOW_INSECURE_TLS=true accepts an unverified connection and serves traffic."""
        with deploy_storage_service(
            name=INSECURE_DEPLOYMENT,
            env={
                ENV_STORAGE_FORMAT: STORAGE_FORMAT_POSTGRES,
                ENV_ALLOW_INSECURE_TLS: "true",
            },
            db_credentials_secret_name=postgres_db_credentials_secret.name,
        ):
            storage_client = StorageBackendClient(
                client=admin_client,
                namespace=model_namespace.name,
                name=INSECURE_DEPLOYMENT,
            )

            readiness = storage_client.readiness()
            assert readiness.status_code == HTTPStatus.OK, f"Readiness failed after opt-in: {readiness.text}"
            checks = {check["name"]: check["status"] for check in readiness.json()["checks"]}
            assert checks.get(STORAGE_CHECK_NAME) == HEALTH_STATUS_OK, (
                f"Storage unready after the insecure opt-in: {readiness.json()}"
            )

            model_name = "postgres-insecure-tls-model"
            n_rows = 2
            upload = storage_client.upload(payload=build_upload_payload(model_name=model_name, n_rows=n_rows))
            assert upload.status_code == HTTPStatus.OK, f"Upload failed over the unverified connection: {upload.text}"

            model_info = wait_for_model_observations(
                storage_client=storage_client, model_name=model_name, expected=n_rows
            )
            assert model_info["data"]["observations"] == n_rows
