from typing import Generator, Any, Callable, Dict
import pytest
import os
import secrets
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.service import Service


MILVUS_IMAGE = os.getenv(
    "LLS_VECTOR_IO_MILVUS_IMAGE",
    "docker.io/milvusdb/milvus@sha256:3d772c3eae3a6107b778636cea5715b9353360b92e5dcfdcaf4ca7022f4f497c",  # Milvus 2.6.3
)
MILVUS_TOKEN = os.getenv("LLS_VECTOR_IO_MILVUS_TOKEN", secrets.token_urlsafe(32))
ETCD_IMAGE = os.getenv(
    "LLS_VECTOR_IO_ETCD_IMAGE",
    "quay.io/coreos/etcd@sha256:3397341272b9e0a6f44d7e3fc7c321c6efe6cbe82ce866b9b01d0c704bfc5bf3",  # etcd v3.6.5
)


@pytest.fixture(scope="class")
def vector_io_provider_deployment_config_factory(
    request: FixtureRequest,
) -> Callable[[str], list[Dict[str, str]]]:
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

    Example:
        def test_with_milvus(vector_io_provider_deployment_config_factory):
            env_vars = vector_io_provider_deployment_config_factory("milvus-remote")
            # env_vars contains MILVUS_ENDPOINT, MILVUS_TOKEN, etc.
    """

    def _factory(provider_name: str) -> list[Dict[str, str]]:
        env_vars: list[dict[str, str]] = []

        if provider_name is None or provider_name == "milvus":
            # Default case - no additional environment variables needed
            pass
        elif provider_name == "milvus-remote":
            request.getfixturevalue(argname="milvus_service")
            env_vars.append({"name": "MILVUS_ENDPOINT", "value": "http://vector-io-milvus-service:19530"})
            env_vars.append({"name": "MILVUS_TOKEN", "value": MILVUS_TOKEN})
            env_vars.append({"name": "MILVUS_CONSISTENCY_LEVEL", "value": "Bounded"})
        elif provider_name == "faiss":
            env_vars.append({"name": "ENABLE_FAISS", "value": "faiss"})
            env_vars.append({
                "name": "FAISS_KVSTORE_DB_PATH",
                "value": "/opt/app-root/src/.llama/distributions/rh/sqlite_vec.db",
            })

        return env_vars

    return _factory


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
) -> Generator[Deployment, Any, Any]:
    """Deploy a remote Milvus instance for vector I/O provider testing."""
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
