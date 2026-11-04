"""Verify the CR selects PostgreSQL through its database Secret and wires TLS."""

import base64

import pytest
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.trustyai_service import TrustyAIService

from tests.ai_safety.trustyai_service.storage_backends.constants import (
    DATABASE_STORAGE_FORMAT,
    ENV_DATABASE_KIND,
    ENV_STORAGE_FORMAT,
    POSTGRES_DATABASE_KIND,
)
from utilities.constants import TRUSTYAI_SERVICE_NAME

DATABASE_SECRET_KEYS: dict[str, str] = {
    "QUARKUS_DATASOURCE_DB_KIND": "databaseKind",
    "QUARKUS_DATASOURCE_USERNAME": "databaseUsername",
    "QUARKUS_DATASOURCE_PASSWORD": "databasePassword",  # pragma: allowlist secret
    "DATABASE_SERVICE": "databaseService",
    "DATABASE_PORT": "databasePort",
    "DATABASE_NAME": "databaseName",
    "QUARKUS_DATASOURCE_GENERATION": "databaseGeneration",
}
DB_CA_MOUNT_PATH: str = "/etc/tls/db"


@pytest.mark.ai_safety
@pytest.mark.tier2
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-trustyai-database-configuration"})],
    indirect=True,
)
class TestTrustyAIServiceDatabaseConfiguration:
    """The operator's CR contract wires database settings from the referenced Secret."""

    def test_database_secret_and_tls_ca_are_wired_by_the_operator(
        self,
        model_namespace: Namespace,
        database_backed_service: Deployment,
        postgres_db_credentials_secret,
        admin_client,
    ) -> None:
        service = TrustyAIService(
            client=admin_client,
            namespace=model_namespace.name,
            name=TRUSTYAI_SERVICE_NAME,
            ensure_exists=True,
        )
        spec = service.instance.spec
        assert spec.storage.format == DATABASE_STORAGE_FORMAT
        secret_name = spec.storage.databaseConfigurations
        assert secret_name, "The DATABASE storage CR must reference a database configuration Secret"
        assert secret_name == postgres_db_credentials_secret.name

        service_container = next(
            container
            for container in database_backed_service.instance.spec.template.spec.containers
            if container.name == TRUSTYAI_SERVICE_NAME
        )
        env = {entry.name: entry for entry in service_container.env}
        assert env[ENV_STORAGE_FORMAT].value == DATABASE_STORAGE_FORMAT

        for env_name, secret_key in DATABASE_SECRET_KEYS.items():
            entry = env[env_name]
            secret_key_ref = entry.valueFrom.secretKeyRef
            assert secret_key_ref.name == secret_name, f"{env_name} uses a different Secret: {secret_key_ref}"
            assert secret_key_ref.key == secret_key, f"{env_name} reads the wrong Secret key: {secret_key_ref}"

        selected_kind = base64.b64decode(postgres_db_credentials_secret.instance.data["databaseKind"]).decode()
        assert selected_kind == POSTGRES_DATABASE_KIND
        assert env[ENV_DATABASE_KIND].valueFrom.secretKeyRef.key == "databaseKind"

        jdbc_url = env["QUARKUS_DATASOURCE_JDBC_URL"].value
        assert "requireSSL=true" in jdbc_url and "sslMode=verify-ca" in jdbc_url, (
            f"CR-managed database connection does not require TLS verification: {jdbc_url}"
        )
        assert f"{DB_CA_MOUNT_PATH}/ca.crt" in jdbc_url

        pod_spec = database_backed_service.instance.spec.template.spec
        ca_volume = next(volume for volume in pod_spec.volumes if volume.name == "db-ca-cert")
        assert ca_volume.secret.secretName == f"{TRUSTYAI_SERVICE_NAME}-db-ca"
        ca_mount = next(mount for mount in service_container.volumeMounts if mount.name == "db-ca-cert")
        assert ca_mount.mountPath == DB_CA_MOUNT_PATH

        assert model_namespace.name == service.namespace
