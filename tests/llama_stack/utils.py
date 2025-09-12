from contextlib import contextmanager
from typing import Dict, Any, Generator

from kubernetes.dynamic import DynamicClient
from llama_stack_client import LlamaStackClient, APIConnectionError
from ocp_resources.llama_stack_distribution import LlamaStackDistribution
from simple_logger.logger import get_logger
from timeout_sampler import retry

from utilities.constants import Timeout


LOGGER = get_logger(name=__name__)


@contextmanager
def create_llama_stack_distribution(
    client: DynamicClient,
    name: str,
    namespace: str,
    replicas: int,
    server: Dict[str, Any],
    teardown: bool = True,
) -> Generator[LlamaStackDistribution, Any, Any]:
    """
    Context manager to create and optionally delete a LLama Stack Distribution
    """
    with LlamaStackDistribution(
        client=client,
        name=name,
        namespace=namespace,
        replicas=replicas,
        server=server,
        wait_for_resource=True,
        teardown=teardown,
    ) as llama_stack_distribution:
        yield llama_stack_distribution


@retry(wait_timeout=Timeout.TIMEOUT_1MIN, sleep=5)
def wait_for_llama_stack_client_ready(client: LlamaStackClient) -> bool:
    try:
        client.inspect.health()
        version = client.inspect.version()
        LOGGER.info(f"Llama Stack server (v{version.version}) is available!")
        return True
    except APIConnectionError as e:
        LOGGER.debug(f"Llama Stack server not ready yet: {e}")
        return False
    except Exception as e:
        LOGGER.warning(f"Unexpected error checking Llama Stack readiness: {e}")
        return False


def get_milvus_deployment_template() -> dict[str, Any]:
    return {
        "metadata": {"labels": {"app": "milvus-standalone"}},
        "spec": {
            "containers": [
                {
                    "name": "milvus-standalone",
                    "image": "quay.io/mcampbel/milvus:v2.6.0",  # TODO: Replace this image
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
                        {"name": "ETCD_ENDPOINTS", "value": "rag-etcd-service:2379"},
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


def get_etcd_deployment_template() -> dict[str, Any]:
    return {
        "metadata": {"labels": {"app": "etcd"}},
        "spec": {
            "containers": [
                {
                    "name": "etcd",
                    "image": "quay.io/coreos/etcd:v3.5.5",
                    "command": [
                        "etcd",
                        "--advertise-client-urls=http://127.0.0.1:2379",
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
