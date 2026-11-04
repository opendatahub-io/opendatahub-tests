"""PostgreSQL TLS behavior through TrustyAIService CRs."""

from collections.abc import Callable
from contextlib import AbstractContextManager
from http import HTTPStatus

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret

from tests.ai_safety.trustyai_service.storage_backends.constants import (
    ENV_ALLOW_INSECURE_TLS,
    ENV_TLS_CA_CERT,
    HEALTH_STATUS_OK,
    STORAGE_CHECK_NAME,
)
from tests.ai_safety.trustyai_service.storage_backends.test_postgres_storage import POSTGRES_SESSION_PROBE
from tests.ai_safety.trustyai_service.storage_backends.utils import (
    StorageBackendClient,
    build_upload_payload,
    create_database_backed_trustyai_service,
    deployment_pod_selector,
    patch_service_deployment,
    run_python_in_service_pod,
    wait_for_model_observations,
    wait_for_pod_log_message,
)
from tests.ai_safety.trustyai_service.utils import generate_db_tls_certs

NO_CA_SERVICE: str = "trustyai-service-no-ca"
WRONG_CA_SERVICE: str = "trustyai-service-wrong-ca"
MISSING_CA_PATH_SERVICE: str = "trustyai-service-missing-ca"
INSECURE_SERVICE: str = "trustyai-service-insecure"
TLS_VERIFICATION_ERROR: str = "certificate verify failed"
TLS_CA_ERROR: str = "requires a usable CA certificate at"


def _client(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    service_name: str,
    token: str,
) -> StorageBackendClient:
    return StorageBackendClient(
        client=admin_client,
        namespace=model_namespace.name,
        name=service_name,
        token=token,
    )


@pytest.mark.ai_safety
@pytest.mark.tier3
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-trustyai-postgres-tls"})],
    indirect=True,
)
class TestPostgresStorageTls:
    """The CR mounts a CA for verify-full; absent CA follows the service's permissive default."""

    def test_cr_mounted_ca_connects_with_verified_tls(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        postgres_backed_service: Deployment,
    ) -> None:
        """A valid CA and matching PostgreSQL service certificate allow a TLS session."""
        settings = run_python_in_service_pod(
            client=admin_client,
            namespace=model_namespace.name,
            deployment=postgres_backed_service,
            code=POSTGRES_SESSION_PROBE,
        )
        assert settings["ssl"] is True, f"PostgreSQL session is not using TLS: {settings}"

    def test_no_ca_uses_product_default_and_still_serves(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        postgres_db: Deployment,
        postgres_db_credentials_secret: Secret,
        current_client_token: str,
        teardown_resources: bool,
    ) -> None:
        """Unset DATABASE_ALLOW_INSECURE_TLS permits the documented no-CA default."""
        with create_database_backed_trustyai_service(
            client=admin_client,
            namespace=model_namespace.name,
            name=NO_CA_SERVICE,
            credentials_secret_name=postgres_db_credentials_secret.name,
            teardown=teardown_resources,
        ) as (_, deployment):
            client = _client(
                admin_client=admin_client,
                model_namespace=model_namespace,
                service_name=NO_CA_SERVICE,
                token=current_client_token,
            )
            response = client.readiness()
            assert response.status_code == HTTPStatus.OK, f"No-CA default failed readiness: {response.text}"
            checks = {check["name"]: check["status"] for check in response.json()["checks"]}
            assert checks.get(STORAGE_CHECK_NAME) == HEALTH_STATUS_OK, f"Storage is not ready: {response.json()}"
            settings = run_python_in_service_pod(
                client=admin_client,
                namespace=model_namespace.name,
                deployment=deployment,
                code=POSTGRES_SESSION_PROBE,
            )
            assert settings["ssl"] is True, f"PostgreSQL did not negotiate TLS: {settings}"

    def test_wrong_ca_is_rejected(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        postgres_db: Deployment,
        postgres_db_credentials_secret: Secret,
        teardown_resources: bool,
    ) -> None:
        """A CA from a different issuer cannot authenticate the PostgreSQL server."""
        wrong_ca, _, _ = generate_db_tls_certs(
            namespace_name=model_namespace.name,
            service_name="untrusted-postgres",
        )
        with (
            Secret(
                client=admin_client,
                name=f"{WRONG_CA_SERVICE}-db-ca",
                namespace=model_namespace.name,
                string_data={"ca.crt": wrong_ca},
                teardown=teardown_resources,
            ),
            create_database_backed_trustyai_service(
                client=admin_client,
                namespace=model_namespace.name,
                name=WRONG_CA_SERVICE,
                credentials_secret_name=postgres_db_credentials_secret.name,
                wait_for_replicas=False,
                teardown=teardown_resources,
            ) as (_, deployment),
        ):
            logs = wait_for_pod_log_message(
                client=admin_client,
                namespace=model_namespace.name,
                label_selector=deployment_pod_selector(deployment),
                message=TLS_VERIFICATION_ERROR,
            )
            assert TLS_VERIFICATION_ERROR in logs.lower(), f"Wrong CA was not rejected; logs: {logs[-2000:]}"
            current = Deployment(client=admin_client, namespace=model_namespace.name, name=WRONG_CA_SERVICE)
            assert not current.instance.status.get("readyReplicas"), "Service became ready with an untrusted CA"

    def test_missing_configured_ca_is_rejected_when_tls_is_required(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        postgres_db: Deployment,
        postgres_db_credentials_secret: Secret,
        postgres_service_ca_secret: Callable[[str], AbstractContextManager[Secret]],
        teardown_resources: bool,
    ) -> None:
        """The service reports the configured CA path when the caller requires TLS verification."""
        absent_path = "/etc/tls/db/does-not-exist.crt"
        with (
            postgres_service_ca_secret(MISSING_CA_PATH_SERVICE),
            create_database_backed_trustyai_service(
                client=admin_client,
                namespace=model_namespace.name,
                name=MISSING_CA_PATH_SERVICE,
                credentials_secret_name=postgres_db_credentials_secret.name,
                wait_for_replicas=False,
                teardown=teardown_resources,
            ) as (_, deployment),
        ):
            patch_service_deployment(
                deployment=deployment,
                env={ENV_TLS_CA_CERT: absent_path, ENV_ALLOW_INSECURE_TLS: "false"},
                wait_for_replicas=False,
            )
            logs = wait_for_pod_log_message(
                client=admin_client,
                namespace=model_namespace.name,
                label_selector=deployment_pod_selector(deployment),
                message=TLS_CA_ERROR,
            )
            assert TLS_CA_ERROR in logs and absent_path in logs, f"Unexpected TLS diagnostic: {logs[-2000:]}"
            current = Deployment(client=admin_client, namespace=model_namespace.name, name=MISSING_CA_PATH_SERVICE)
            assert not current.instance.status.get("readyReplicas"), "Service became ready without a usable CA"

    def test_insecure_opt_in_allows_startup(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        postgres_db: Deployment,
        postgres_db_credentials_secret: Secret,
        current_client_token: str,
        teardown_resources: bool,
    ) -> None:
        """The explicit insecure opt-in allows PostgreSQL when no CA is mounted."""
        with create_database_backed_trustyai_service(
            client=admin_client,
            namespace=model_namespace.name,
            name=INSECURE_SERVICE,
            credentials_secret_name=postgres_db_credentials_secret.name,
            teardown=teardown_resources,
        ) as (_, deployment):
            patch_service_deployment(deployment=deployment, env={ENV_ALLOW_INSECURE_TLS: "true"})
            client = _client(
                admin_client=admin_client,
                model_namespace=model_namespace,
                service_name=INSECURE_SERVICE,
                token=current_client_token,
            )
            readiness = client.readiness()
            assert readiness.status_code == HTTPStatus.OK, f"Readiness failed after opt-in: {readiness.text}"
            checks = {check["name"]: check["status"] for check in readiness.json()["checks"]}
            assert checks.get(STORAGE_CHECK_NAME) == HEALTH_STATUS_OK, (
                f"Storage unready after insecure opt-in: {readiness.json()}"
            )

            model_name = "postgres-insecure-tls-model"
            n_rows = 2
            upload = client.upload(payload=build_upload_payload(model_name=model_name, n_rows=n_rows))
            assert upload.status_code == HTTPStatus.OK, f"Upload failed: {upload.text}"
            model_info = wait_for_model_observations(storage_client=client, model_name=model_name, expected=n_rows)
            assert model_info["data"]["observations"] == n_rows
