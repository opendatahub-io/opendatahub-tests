from typing import Generator, Any, Callable, Dict
import pytest
import os
import secrets
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.service import Service
from ocp_resources.secret import Secret


MILVUS_IMAGE = os.getenv(
    "LLS_VECTOR_IO_MILVUS_IMAGE",
    "docker.io/milvusdb/milvus@sha256:3d772c3eae3a6107b778636cea5715b9353360b92e5dcfdcaf4ca7022f4f497c",  # Milvus 2.6.3
)
MILVUS_TOKEN = os.getenv("LLS_VECTOR_IO_MILVUS_TOKEN", secrets.token_urlsafe(32))
ETCD_IMAGE = os.getenv(
    "LLS_VECTOR_IO_ETCD_IMAGE",
    "quay.io/coreos/etcd@sha256:3397341272b9e0a6f44d7e3fc7c321c6efe6cbe82ce866b9b01d0c704bfc5bf3",  # etcd v3.6.5
)

PGVECTOR_IMAGE = os.getenv(
    "LLS_VECTOR_IO_PGVECTOR_IMAGE",
    (
        "docker.io/pgvector/pgvector@sha256:"
        "0a07c4114ba6d1d04effcce3385e9f5ce305eb02e56a3d35948a415a52f193ec"  # pgvector 16  # pragma: allowlist secret
    ),
)

PGVECTOR_USER = os.getenv("LLS_VECTOR_IO_PGVECTOR_USER", "vector_user")
PGVECTOR_PASSWORD = os.getenv("LLS_VECTOR_IO_PGVECTOR_PASSWORD", "yourpassword")

# qdrant v1 unprivileged latest
QDRANT_IMAGE = os.getenv(
    "LLS_VECTOR_IO_QDRANT_IMAGE",
    (
        "docker.io/qdrant/qdrant@sha256:"
        "9dfabc51ededc48158899a288a19a04de1ab54a11d8c512e1c40eebbd5e2bc92"  # pragma: allowlist secret
    ),
)

QDRANT_API_KEY = os.getenv("LLS_VECTOR_IO_QDRANT_API_KEY", "yourapikey")
QDRANT_URL = os.getenv("LLS_VECTOR_IO_QDRANT_URL", "http://vector-io-qdrant-service:6333")


@pytest.fixture(scope="class")
def vector_io_provider_deployment_config_factory(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    vector_io_secret: Secret,
    request: FixtureRequest,
) -> Callable[[str], list[Dict[str, Any]]]:
    """
    Factory fixture for deploying vector I/O providers and returning their configuration.

    This fixture returns a factory function that can deploy different vector I/O providers
    (such as Milvus) in the cluster and return the necessary environment variables
    for configuring the LlamaStack server to use these providers.

    Args:
        request: Pytest fixture request object for accessing other fixtures

    Returns:
        Callable[[str], list[Dict[str, str]]]: Factory function that takes a provider name
        and returns a list of environment variable dictionaries

    Supported Providers:
        - "milvus" (or None): Local Milvus instance with embedded database
        - "milvus-remote": Remote Milvus service requiring external deployment

    Environment Variables by Provider:
        - "milvus": no env vars available
        - "milvus-remote":
          * MILVUS_ENDPOINT: Remote Milvus service endpoint URL
          * MILVUS_TOKEN: Authentication token for remote service
          * MILVUS_CONSISTENCY_LEVEL: Consistency level for operations
        - "pgvector":
          * ENABLE_PGVECTOR: Enable pgvector provider
          * PGVECTOR_HOST: PGVector service hostname
          * PGVECTOR_PORT: PGVector port
          * PGVECTOR_USER: Database user
          * PGVECTOR_PASSWORD: Database password
          * PGVECTOR_DB: Database name
        - "qdrant-remote":
          * ENABLE_QDRANT: enable qdrant provider
          * QDRANT_API_KEY: Qdrant API key
          * QDRANT_URL: Qdrant service URL with protocol (e.g., "http://vector-io-qdrant-service:6333")

    Example:
        def test_with_milvus(vector_io_provider_deployment_config_factory):
            env_vars = vector_io_provider_deployment_config_factory("milvus-remote")
            # env_vars contains MILVUS_ENDPOINT, MILVUS_TOKEN, etc.
    """
    _ = unprivileged_client
    _ = unprivileged_model_namespace
    _ = vector_io_secret

    def _factory(provider_name: str) -> list[Dict[str, Any]]:
        env_vars: list[dict[str, Any]] = []

        if provider_name is None or provider_name == "milvus":
            # Default case - no additional environment variables needed
            pass
        elif provider_name == "milvus-remote":
            request.getfixturevalue(argname="milvus_service")
            env_vars.append({"name": "MILVUS_ENDPOINT", "value": "http://vector-io-milvus-service:19530"})
            env_vars.append(
                {
                    "name": "MILVUS_TOKEN",
                    "valueFrom": {"secretKeyRef": {"name": "vector-io-secret", "key": "milvus-token"}},
                },
            )
            env_vars.append({"name": "MILVUS_CONSISTENCY_LEVEL", "value": "Bounded"})
        elif provider_name == "faiss":
            env_vars.append({"name": "ENABLE_FAISS", "value": "faiss"})
            env_vars.append({
                "name": "FAISS_KVSTORE_DB_PATH",
                "value": "/opt/app-root/src/.llama/distributions/rh/sqlite_vec.db",
            })
        elif provider_name == "pgvector":
            request.getfixturevalue(argname="pgvector_service")
            env_vars.append({"name": "ENABLE_PGVECTOR", "value": "true"})
            env_vars.append({"name": "PGVECTOR_HOST", "value": "vector-io-pgvector-service"})
            env_vars.append({"name": "PGVECTOR_PORT", "value": "5432"})
            env_vars.append(
                {
                    "name": "PGVECTOR_USER",
                    "valueFrom": {"secretKeyRef": {"name": "vector-io-secret", "key": "pgvector-user"}},
                },
            )
            env_vars.append(
                {
                    "name": "PGVECTOR_PASSWORD",
                    "valueFrom": {"secretKeyRef": {"name": "vector-io-secret", "key": "pgvector-password"}},
                },
            )
            env_vars.append({"name": "PGVECTOR_DB", "value": "pgvector"})
        elif provider_name == "qdrant-remote":
            request.getfixturevalue(argname="qdrant_service")
            env_vars.append({"name": "ENABLE_QDRANT", "value": "true"})
            env_vars.append({"name": "QDRANT_URL", "value": QDRANT_URL})
            env_vars.append({
                "name": "QDRANT_API_KEY",
                "valueFrom": {"secretKeyRef": {"name": "vector-io-secret", "key": "qdrant-api-key"}},
            })

        return env_vars

    return _factory


@pytest.fixture(scope="class")
def vector_io_secret(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[Secret, Any, Any]:
    """Create a secret for the vector I/O providers"""
    with Secret(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        name="vector-io-secret",
        type="Opaque",
        string_data={
            "qdrant-api-key": QDRANT_API_KEY,
            "pgvector-user": PGVECTOR_USER,
            "pgvector-password": PGVECTOR_PASSWORD,
            "milvus-token": MILVUS_TOKEN,
        },
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def etcd_deployment(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[Deployment, Any, Any]:
    """Deploy an etcd instance for vector I/O provider testing."""
    with Deployment(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        name="vector-io-etcd-deployment",
        replicas=1,
        selector={"matchLabels": {"app": "etcd"}},
        strategy={"type": "Recreate"},
        template=get_etcd_deployment_template(),
        teardown=True,
    ) as deployment:
        deployment.wait_for_replicas(deployed=True, timeout=120)
        yield deployment


@pytest.fixture(scope="class")
def etcd_service(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[Service, Any, Any]:
    """Create a service for the etcd deployment."""
    with Service(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        name="vector-io-etcd-service",
        ports=[
            {
                "port": 2379,
                "targetPort": 2379,
            }
        ],
        selector={"app": "etcd"},
        wait_for_resource=True,
    ) as service:
        yield service


@pytest.fixture(scope="class")
def remote_milvus_deployment(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    etcd_deployment: Deployment,
    etcd_service: Service,
    vector_io_secret: Secret,
) -> Generator[Deployment, Any, Any]:
    """Deploy a remote Milvus instance for vector I/O provider testing."""
    _ = etcd_deployment
    _ = etcd_service
    _ = vector_io_secret

    with Deployment(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        name="vector-io-milvus-deployment",
        min_ready_seconds=5,
        replicas=1,
        selector={"matchLabels": {"app": "milvus-standalone"}},
        strategy={"type": "Recreate"},
        template=get_milvus_deployment_template(),
        teardown=True,
    ) as deployment:
        deployment.wait_for_replicas(deployed=True, timeout=240)
        yield deployment


@pytest.fixture(scope="class")
def milvus_service(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    remote_milvus_deployment: Deployment,
) -> Generator[Service, Any, Any]:
    """Create a service for the remote Milvus deployment."""
    _ = remote_milvus_deployment

    with Service(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        name="vector-io-milvus-service",
        ports=[
            {
                "name": "grpc",
                "port": 19530,
                "targetPort": 19530,
            },
        ],
        selector={"app": "milvus-standalone"},
        wait_for_resource=True,
    ) as service:
        yield service


def get_milvus_deployment_template() -> Dict[str, Any]:
    """Return the Kubernetes deployment template for Milvus standalone."""
    return {
        "metadata": {"labels": {"app": "milvus-standalone"}},
        "spec": {
            "containers": [
                {
                    "name": "milvus-standalone",
                    "image": MILVUS_IMAGE,
                    "args": ["milvus", "run", "standalone"],
                    "ports": [{"containerPort": 19530, "protocol": "TCP"}],
                    "volumeMounts": [
                        {
                            "name": "milvus-data",
                            "mountPath": "/var/lib/milvus",
                        }
                    ],
                    "env": [
                        {"name": "DEPLOY_MODE", "value": "standalone"},
                        {"name": "ETCD_ENDPOINTS", "value": "vector-io-etcd-service:2379"},
                        {"name": "MINIO_ADDRESS", "value": ""},
                        {"name": "COMMON_STORAGETYPE", "value": "local"},
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "milvus-data",
                    "emptyDir": {},
                }
            ],
        },
    }


def get_etcd_deployment_template() -> Dict[str, Any]:
    """Return the Kubernetes deployment template for etcd."""
    return {
        "metadata": {"labels": {"app": "etcd"}},
        "spec": {
            "containers": [
                {
                    "name": "etcd",
                    "image": ETCD_IMAGE,
                    "command": [
                        "etcd",
                        "--advertise-client-urls=http://vector-io-etcd-service:2379",
                        "--listen-client-urls=http://0.0.0.0:2379",
                        "--data-dir=/etcd",
                    ],
                    "ports": [{"containerPort": 2379}],
                    "volumeMounts": [
                        {
                            "name": "etcd-data",
                            "mountPath": "/etcd",
                        }
                    ],
                    "env": [
                        {"name": "ETCD_AUTO_COMPACTION_MODE", "value": "revision"},
                        {"name": "ETCD_AUTO_COMPACTION_RETENTION", "value": "1000"},
                        {"name": "ETCD_QUOTA_BACKEND_BYTES", "value": "4294967296"},
                        {"name": "ETCD_SNAPSHOT_COUNT", "value": "50000"},
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "etcd-data",
                    "emptyDir": {},
                }
            ],
        },
    }


@pytest.fixture(scope="class")
def pgvector_deployment(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    vector_io_secret: Secret,
) -> Generator[Deployment, Any, Any]:
    """Deploy a PGVector instance for vector I/O provider testing."""
    _ = vector_io_secret

    with Deployment(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        name="vector-io-pgvector-deployment",
        min_ready_seconds=5,
        replicas=1,
        selector={"matchLabels": {"app": "pgvector"}},
        strategy={"type": "Recreate"},
        template=get_pgvector_deployment_template(),
        teardown=True,
    ) as deployment:
        deployment.wait_for_replicas(deployed=True, timeout=240)
        yield deployment


@pytest.fixture(scope="class")
def pgvector_service(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    pgvector_deployment: Deployment,
) -> Generator[Service, Any, Any]:
    """Create a service for the PGVector deployment."""
    _ = pgvector_deployment

    with Service(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        name="vector-io-pgvector-service",
        ports=[
            {
                "name": "postgres",
                "port": 5432,
                "targetPort": 5432,
            },
        ],
        selector={"app": "pgvector"},
        wait_for_resource=True,
    ) as service:
        yield service


def get_pgvector_deployment_template() -> Dict[str, Any]:
    """Return a Kubernetes deployment for PGVector"""
    return {
        "metadata": {"labels": {"app": "pgvector"}},
        "spec": {
            "containers": [
                {
                    "name": "pgvector",
                    "image": PGVECTOR_IMAGE,
                    "ports": [{"containerPort": 5432}],
                    "env": [
                        {"name": "POSTGRES_DB", "value": "pgvector"},
                        {
                            "name": "POSTGRES_USER",
                            "valueFrom": {"secretKeyRef": {"name": "vector-io-secret", "key": "pgvector-user"}},
                        },
                        {
                            "name": "POSTGRES_PASSWORD",
                            "valueFrom": {"secretKeyRef": {"name": "vector-io-secret", "key": "pgvector-password"}},
                        },
                        {"name": "PGDATA", "value": "/var/lib/postgresql/data/pgdata"},
                    ],
                    "lifecycle": {
                        "postStart": {
                            "exec": {
                                "command": [
                                    "/bin/sh",
                                    "-c",
                                    (
                                        "sleep 5\n"
                                        f"PGPASSWORD={PGVECTOR_PASSWORD} psql -h localhost -U {PGVECTOR_USER} "
                                        '-d pgvector -c "CREATE EXTENSION IF NOT EXISTS vector;" || true'
                                    ),
                                ]
                            }
                        }
                    },
                    "volumeMounts": [{"name": "pgdata", "mountPath": "/var/lib/postgresql/data"}],
                }
            ],
            "volumes": [{"name": "pgdata", "emptyDir": {}}],
        },
    }


@pytest.fixture(scope="class")
def qdrant_deployment(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    vector_io_secret: Secret,
) -> Generator[Deployment, Any, Any]:
    """Deploy a Qdrant instance for vector I/O provider testing."""
    _ = vector_io_secret

    with Deployment(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        name="vector-io-qdrant-deployment",
        min_ready_seconds=5,
        replicas=1,
        selector={"matchLabels": {"app": "qdrant"}},
        strategy={"type": "Recreate"},
        template=get_qdrant_deployment_template(),
        teardown=True,
    ) as deployment:
        deployment.wait_for_replicas(deployed=True, timeout=240)
        yield deployment


@pytest.fixture(scope="class")
def qdrant_service(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    qdrant_deployment: Deployment,
) -> Generator[Service, Any, Any]:
    """Create a service for the Qdrant deployment."""
    _ = qdrant_deployment

    with Service(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        name="vector-io-qdrant-service",
        ports=[
            {
                "name": "http",
                "port": 6333,
                "targetPort": 6333,
            },
            {
                "name": "grpc",
                "port": 6334,
                "targetPort": 6334,
            },
        ],
        selector={"app": "qdrant"},
        wait_for_resource=True,
    ) as service:
        yield service


def get_qdrant_deployment_template() -> Dict[str, Any]:
    """Return a Kubernetes deployment for Qdrant"""
    return {
        "metadata": {"labels": {"app": "qdrant"}},
        "spec": {
            "containers": [
                {
                    "name": "qdrant",
                    "image": QDRANT_IMAGE,
                    "ports": [
                        {
                            "containerPort": 6333,
                            "name": "http",
                        },
                        {
                            "containerPort": 6334,
                            "name": "grpc",
                        },
                    ],
                    "env": [
                        {
                            "name": "QDRANT__SERVICE__API_KEY",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": "vector-io-secret",
                                    "key": "qdrant-api-key",
                                },
                            },
                        },
                    ],
                    "volumeMounts": [
                        {"name": "qdrant-storage", "mountPath": "/qdrant/storage"},
                        {
                            "name": "qdrant-storage",
                            "mountPath": "/qdrant/snapshots",
                            "subPath": "snapshots",
                        },
                    ],
                },
            ],
            "volumes": [{"name": "qdrant-storage", "emptyDir": {}}],
        },
    }
