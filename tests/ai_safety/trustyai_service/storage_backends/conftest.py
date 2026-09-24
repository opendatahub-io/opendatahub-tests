"""Fixtures for the TrustyAI SQL storage backend tests."""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret

from tests.ai_safety.image_constants import AiSafetyImages
from tests.ai_safety.trustyai_service.storage_backends.constants import (
    ENV_ALLOW_INSECURE_TLS,
    ENV_SQLITE_PATH,
    ENV_STORAGE_FORMAT,
    POSTGRES_CA_SECRET_NAME,
    POSTGRES_CREDENTIALS_SECRET_NAME,
    POSTGRES_DB_NAME,
    POSTGRES_DB_PASSWORD,
    POSTGRES_DB_USERNAME,
    POSTGRES_PORT,
    POSTGRES_SERVICE_DEPLOYMENT,
    POSTGRES_SERVICE_NAME,
    SERVICE_DB_CA_MOUNT_PATH,
    SQLITE_DATA_FILE,
    SQLITE_DATA_MOUNT_PATH,
    SQLITE_MEMORY_PATH,
    SQLITE_MEMORY_SERVICE_DEPLOYMENT,
    SQLITE_SERVICE_DEPLOYMENT,
    STORAGE_FORMAT_POSTGRES,
    STORAGE_FORMAT_SQLITE,
)
from tests.ai_safety.trustyai_service.storage_backends.utils import (
    StorageBackendClient,
    create_standalone_postgres,
    create_storage_backend_service,
)

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="class")
def trustyai_service_image() -> str:
    """Candidate service image with PostgreSQL/SQLite/SQLAlchemy support (RHAISTRAT-2662)."""
    image = AiSafetyImages.TRUSTYAI_SERVICE_SQL_BACKENDS
    LOGGER.info(f"TrustyAI service image under test: {image}")
    return image


@pytest.fixture(scope="class")
def postgres_db_credentials_secret(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[Secret, Any, Any]:
    """Credentials shared by the PostgreSQL deployment and the service under test."""
    with Secret(
        client=admin_client,
        name=POSTGRES_CREDENTIALS_SECRET_NAME,
        namespace=model_namespace.name,
        string_data={
            "databaseKind": "postgresql",
            "databaseName": POSTGRES_DB_NAME,
            "databaseUsername": POSTGRES_DB_USERNAME,
            "databasePassword": POSTGRES_DB_PASSWORD,
            "databaseService": POSTGRES_SERVICE_NAME,
            "databasePort": str(POSTGRES_PORT),
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
    """A TLS-enabled PostgreSQL deployment in the test namespace."""
    with create_standalone_postgres(
        client=admin_client,
        namespace_name=model_namespace.name,
        name=POSTGRES_SERVICE_NAME,
        db_credentials_secret_name=postgres_db_credentials_secret.name,
        teardown=teardown_resources,
    ) as deployment:
        yield deployment


@pytest.fixture(scope="class")
def postgres_db_ca_secret(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    postgres_db: Deployment,
    teardown_resources: bool,
) -> Generator[Secret, Any, Any]:
    """Copy of the database CA, mounted into the service so it can verify the server."""
    db_ca_secret = Secret(
        client=admin_client,
        name=f"{POSTGRES_SERVICE_NAME}-ca",
        namespace=model_namespace.name,
        ensure_exists=True,
    )
    with Secret(
        client=admin_client,
        name=POSTGRES_CA_SECRET_NAME,
        namespace=model_namespace.name,
        data_dict={"ca.crt": db_ca_secret.instance.data["ca.crt"]},
        teardown=teardown_resources,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def deploy_storage_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    trustyai_service_image: str,
    teardown_resources: bool,
) -> Callable[..., Any]:
    """Factory that deploys the service image with caller-supplied env.

    Used by tests that need a one-off configuration (a missing CA, an insecure
    opt-in, custom pool sizing) rather than one of the ready-made fixtures.
    """

    @contextmanager
    def _deploy(
        name: str,
        env: dict[str, str],
        db_credentials_secret_name: str | None = None,
        ca_secret_name: str | None = None,
        ca_mount_path: str | None = None,
        data_volume_mount_path: str | None = None,
        wait_for_ready: bool = True,
    ) -> Generator[Deployment, Any, Any]:
        with create_storage_backend_service(
            client=admin_client,
            namespace_name=model_namespace.name,
            name=name,
            image=trustyai_service_image,
            env=env,
            db_credentials_secret_name=db_credentials_secret_name,
            ca_secret_name=ca_secret_name,
            ca_mount_path=ca_mount_path,
            data_volume_mount_path=data_volume_mount_path,
            wait_for_ready=wait_for_ready,
            teardown=teardown_resources,
        ) as deployment:
            yield deployment

    return _deploy


@pytest.fixture(scope="class")
def postgres_backed_service(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    trustyai_service_image: str,
    postgres_db: Deployment,
    postgres_db_credentials_secret: Secret,
    postgres_db_ca_secret: Secret,
    teardown_resources: bool,
) -> Generator[Deployment, Any, Any]:
    """TrustyAI service running against PostgreSQL over verified TLS.

    Tests may add or override env vars with
    `@pytest.mark.parametrize("postgres_backed_service", [{"env": {...}}], indirect=True)`.
    """
    extra_env: dict[str, str] = getattr(request, "param", {}).get("env", {})

    with create_storage_backend_service(
        client=admin_client,
        namespace_name=model_namespace.name,
        name=POSTGRES_SERVICE_DEPLOYMENT,
        image=trustyai_service_image,
        env={ENV_STORAGE_FORMAT: STORAGE_FORMAT_POSTGRES, **extra_env},
        db_credentials_secret_name=postgres_db_credentials_secret.name,
        ca_secret_name=postgres_db_ca_secret.name,
        ca_mount_path=SERVICE_DB_CA_MOUNT_PATH,
        teardown=teardown_resources,
    ) as deployment:
        yield deployment


@pytest.fixture(scope="class")
def postgres_storage_client(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    postgres_backed_service: Deployment,
) -> StorageBackendClient:
    """HTTP client for the PostgreSQL-backed service."""
    return StorageBackendClient(
        client=admin_client,
        namespace=model_namespace.name,
        name=POSTGRES_SERVICE_DEPLOYMENT,
    )


@pytest.fixture(scope="class")
def sqlite_backed_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    trustyai_service_image: str,
    teardown_resources: bool,
) -> Generator[Deployment, Any, Any]:
    """TrustyAI service running against a file-backed SQLite database."""
    with create_storage_backend_service(
        client=admin_client,
        namespace_name=model_namespace.name,
        name=SQLITE_SERVICE_DEPLOYMENT,
        image=trustyai_service_image,
        env={
            ENV_STORAGE_FORMAT: STORAGE_FORMAT_SQLITE,
            ENV_SQLITE_PATH: SQLITE_DATA_FILE,
            # SQLite is local to the pod, so there is no database TLS to configure.
            ENV_ALLOW_INSECURE_TLS: "true",
        },
        data_volume_mount_path=SQLITE_DATA_MOUNT_PATH,
        teardown=teardown_resources,
    ) as deployment:
        yield deployment


@pytest.fixture(scope="class")
def sqlite_storage_client(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    sqlite_backed_service: Deployment,
) -> StorageBackendClient:
    """HTTP client for the file-backed SQLite service."""
    return StorageBackendClient(
        client=admin_client,
        namespace=model_namespace.name,
        name=SQLITE_SERVICE_DEPLOYMENT,
    )


@pytest.fixture(scope="class")
def sqlite_memory_backed_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    trustyai_service_image: str,
    teardown_resources: bool,
) -> Generator[Deployment, Any, Any]:
    """TrustyAI service running against an in-memory SQLite database."""
    with create_storage_backend_service(
        client=admin_client,
        namespace_name=model_namespace.name,
        name=SQLITE_MEMORY_SERVICE_DEPLOYMENT,
        image=trustyai_service_image,
        env={
            ENV_STORAGE_FORMAT: STORAGE_FORMAT_SQLITE,
            ENV_SQLITE_PATH: SQLITE_MEMORY_PATH,
            ENV_ALLOW_INSECURE_TLS: "true",
        },
        teardown=teardown_resources,
    ) as deployment:
        yield deployment


@pytest.fixture(scope="class")
def sqlite_memory_storage_client(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    sqlite_memory_backed_service: Deployment,
) -> StorageBackendClient:
    """HTTP client for the in-memory SQLite service."""
    return StorageBackendClient(
        client=admin_client,
        namespace=model_namespace.name,
        name=SQLITE_MEMORY_SERVICE_DEPLOYMENT,
    )
