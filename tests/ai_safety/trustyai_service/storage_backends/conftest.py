"""Fixtures for SQL storage tests using the operator-managed TrustyAIService CR."""

from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret

from tests.ai_safety.trustyai_service.storage_backends.constants import (
    POSTGRES_CREDENTIALS_SECRET_NAME,
    POSTGRES_DATABASE_KIND,
    POSTGRES_DB_NAME,
    POSTGRES_DB_PASSWORD,
    POSTGRES_DB_USERNAME,
    POSTGRES_PORT,
    POSTGRES_SERVICE_NAME,
    SQLITE_DATABASE_KIND,
)
from tests.ai_safety.trustyai_service.storage_backends.utils import (
    StorageBackendClient,
    create_database_backed_trustyai_service,
    create_standalone_postgres,
    patch_service_deployment,
)
from utilities.constants import TRUSTYAI_SERVICE_NAME


@pytest.fixture(scope="class")
def postgres_db_credentials_secret(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[Secret, Any, Any]:
    """Credentials and backend selector shared by PostgreSQL and the TrustyAI CR."""
    with Secret(
        client=admin_client,
        name=POSTGRES_CREDENTIALS_SECRET_NAME,
        namespace=model_namespace.name,
        string_data={
            "databaseKind": POSTGRES_DATABASE_KIND,
            "databaseName": POSTGRES_DB_NAME,
            "databaseUsername": POSTGRES_DB_USERNAME,
            "databasePassword": POSTGRES_DB_PASSWORD,
            "databaseService": POSTGRES_SERVICE_NAME,
            "databasePort": str(POSTGRES_PORT),
            "databaseGeneration": "update",
        },
        teardown=teardown_resources,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def postgres_db(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    postgres_db_credentials_secret: Secret,
    teardown_resources: bool,
) -> Generator[Deployment, Any, Any]:
    """A TLS-enabled PostgreSQL database; this is the only directly created Deployment."""
    with create_standalone_postgres(
        client=admin_client,
        namespace_name=model_namespace.name,
        name=POSTGRES_SERVICE_NAME,
        db_credentials_secret_name=postgres_db_credentials_secret.name,
        teardown=teardown_resources,
    ) as deployment:
        yield deployment


@pytest.fixture(scope="class")
def postgres_service_ca_secret(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    postgres_db: Deployment,
    teardown_resources: bool,
) -> Callable[[str], AbstractContextManager[Secret]]:
    """Factory for a DB CA Secret named as the TrustyAIService operator expects."""

    @contextmanager
    def _deploy(service_name: str) -> Generator[Secret, Any, Any]:
        database_ca = Secret(
            client=admin_client,
            name=f"{POSTGRES_SERVICE_NAME}-ca",
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        with Secret(
            client=admin_client,
            name=f"{service_name}-db-ca",
            namespace=model_namespace.name,
            data_dict={"ca.crt": database_ca.instance.data["ca.crt"]},
            teardown=teardown_resources,
        ) as secret:
            yield secret

    return _deploy


@pytest.fixture(scope="class")
def postgres_db_ca_secret(
    postgres_service_ca_secret: Callable[[str], Any],
) -> Generator[Secret, Any, Any]:
    """CA Secret used by the default TrustyAIService CR in this namespace."""
    with postgres_service_ca_secret(TRUSTYAI_SERVICE_NAME) as secret:
        yield secret


@pytest.fixture(scope="class")
def deploy_db_config_map(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    teardown_resources: bool,
) -> Callable[..., AbstractContextManager[ConfigMap]]:
    """Factory for test-provided non-secret SQL settings."""

    @contextmanager
    def _deploy(name: str, data: dict[str, str]) -> Generator[ConfigMap, Any, Any]:
        with ConfigMap(
            client=admin_client,
            name=name,
            namespace=model_namespace.name,
            data=data,
            teardown=teardown_resources,
        ) as config_map:
            yield config_map

    return _deploy


@pytest.fixture(scope="class")
def postgres_backed_service(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    postgres_db: Deployment,
    postgres_db_credentials_secret: Secret,
    postgres_db_ca_secret: Secret,
    teardown_resources: bool,
) -> Generator[Deployment, Any, Any]:
    """PostgreSQL service created by the CR; the operator chooses its image."""
    extra_env: dict[str, str] = getattr(request, "param", {}).get("env", {})
    with create_database_backed_trustyai_service(
        client=admin_client,
        namespace=model_namespace.name,
        name=TRUSTYAI_SERVICE_NAME,
        credentials_secret_name=postgres_db_credentials_secret.name,
        teardown=teardown_resources,
    ) as (_, deployment):
        if extra_env:
            patch_service_deployment(deployment=deployment, env=extra_env)
        yield deployment


@pytest.fixture(scope="class")
def postgres_storage_client(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    postgres_backed_service: Deployment,
    current_client_token: str,
) -> StorageBackendClient:
    """Authenticated client for the operator-created PostgreSQL-backed service Route."""
    return StorageBackendClient(
        client=admin_client,
        namespace=model_namespace.name,
        name=TRUSTYAI_SERVICE_NAME,
        token=current_client_token,
    )


@pytest.fixture(scope="class")
def sqlite_credentials_secret(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[Secret, Any, Any]:
    """Secret selects SQLite through the CR's DATABASE storage configuration."""
    with Secret(
        client=admin_client,
        name="sqlite-db-credentials",
        namespace=model_namespace.name,
        string_data={
            "databaseKind": SQLITE_DATABASE_KIND,
            "databaseName": "trustyai.db",
            "databaseUsername": "unused",
            "databasePassword": "unused",  # pragma: allowlist secret
            "databaseService": "unused",
            "databasePort": str(POSTGRES_PORT),
            "databaseGeneration": "update",
        },
        teardown=teardown_resources,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def sqlite_backed_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    sqlite_credentials_secret: Secret,
    teardown_resources: bool,
) -> Generator[Deployment, Any, Any]:
    """SQLite service using the service's default in-memory database."""
    with create_database_backed_trustyai_service(
        client=admin_client,
        namespace=model_namespace.name,
        name=TRUSTYAI_SERVICE_NAME,
        credentials_secret_name=sqlite_credentials_secret.name,
        teardown=teardown_resources,
    ) as (_, deployment):
        yield deployment


@pytest.fixture(scope="class")
def sqlite_storage_client(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    sqlite_backed_service: Deployment,
    current_client_token: str,
) -> StorageBackendClient:
    return StorageBackendClient(
        client=admin_client,
        namespace=model_namespace.name,
        name=TRUSTYAI_SERVICE_NAME,
        token=current_client_token,
    )


@pytest.fixture(scope="class")
def sqlite_memory_backed_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    sqlite_credentials_secret: Secret,
    teardown_resources: bool,
) -> Generator[Deployment, Any, Any]:
    """A second CR-created SQLite service uses the default in-memory database."""
    service_name = "trustyai-service-memory"
    with create_database_backed_trustyai_service(
        client=admin_client,
        namespace=model_namespace.name,
        name=service_name,
        credentials_secret_name=sqlite_credentials_secret.name,
        teardown=teardown_resources,
    ) as (_, deployment):
        yield deployment


@pytest.fixture(scope="class")
def sqlite_memory_storage_client(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    sqlite_memory_backed_service: Deployment,
    current_client_token: str,
) -> StorageBackendClient:
    return StorageBackendClient(
        client=admin_client,
        namespace=model_namespace.name,
        name="trustyai-service-memory",
        token=current_client_token,
    )


@pytest.fixture(scope="class")
def database_backed_service(
    postgres_backed_service: Deployment,
) -> Deployment:
    """Compatibility fixture for PostgreSQL configuration assertions."""
    return postgres_backed_service
