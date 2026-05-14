import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from llama_stack_client import APIConnectionError, InternalServerError, LlamaStackClient
from ocp_resources.pod import Pod
from timeout_sampler import retry

from utilities.exceptions import UnexpectedResourceCountError
from utilities.resources.llama_stack_distribution import LlamaStackDistribution

LOGGER = structlog.get_logger(name=__name__)

LLS_CORE_POD_FILTER: str = "app=llama-stack"

# False by default to allow testing with self-signed certificates
LLS_CLIENT_VERIFY_SSL: bool = os.getenv("LLS_CLIENT_VERIFY_SSL", "false").lower() == "true"

POSTGRES_IMAGE: str = os.getenv(
    "LLS_VECTOR_IO_POSTGRES_IMAGE",
    (
        "registry.redhat.io/rhel9/postgresql-15@sha256:"
        "90ec347a35ab8a5d530c8d09f5347b13cc71df04f3b994bfa8b1a409b1171d59"  # postgres 15 # pragma: allowlist secret
    ),
)

LLAMA_STACK_DISTRIBUTION_SECRET_DATA: dict[str, str] = {
    "postgres-user": os.getenv("LLS_VECTOR_IO_POSTGRESQL_USER", "ps_user"),
    "postgres-password": os.getenv("LLS_VECTOR_IO_POSTGRESQL_PASSWORD", "ps_password"),
    "vllm-api-token": os.getenv("LLS_CORE_VLLM_API_TOKEN", ""),
    "vllm-embedding-api-token": os.getenv("LLS_CORE_VLLM_EMBEDDING_API_TOKEN", "fake"),
    "aws-access-key-id": os.getenv("AWS_ACCESS_KEY_ID", ""),
    "aws-secret-access-key": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
}


@contextmanager
def create_llama_stack_distribution(
    client: DynamicClient,
    name: str,
    namespace: str,
    replicas: int,
    server: dict[str, Any],
    teardown: bool = True,
) -> Generator[LlamaStackDistribution, Any, Any]:
    # Starting with RHOAI 3.3, pods in the 'openshift-ingress' namespace must be allowed
    # to access the llama-stack-service for the llama_stack_test_route to function properly.
    network: dict[str, Any] = {
        "allowedFrom": {
            "namespaces": ["openshift-ingress"],
        },
    }

    with LlamaStackDistribution(
        client=client,
        name=name,
        namespace=namespace,
        replicas=replicas,
        network=network,
        server=server,
        wait_for_resource=True,
        teardown=teardown,
    ) as llama_stack_distribution:
        yield llama_stack_distribution


@retry(
    wait_timeout=240,
    sleep=5,
    exceptions_dict={ResourceNotFoundError: [], UnexpectedResourceCountError: []},
)
def wait_for_unique_llama_stack_pod(client: DynamicClient, namespace: str) -> Pod:
    """Wait until exactly one LlamaStackDistribution pod is found in the namespace."""
    pods = list(
        Pod.get(
            client=client,
            namespace=namespace,
            label_selector=LLS_CORE_POD_FILTER,
        )
    )
    if not pods:
        raise ResourceNotFoundError(f"No pods found with label selector {LLS_CORE_POD_FILTER} in namespace {namespace}")
    if len(pods) != 1:
        raise UnexpectedResourceCountError(
            f"Expected exactly 1 pod with label selector {LLS_CORE_POD_FILTER} "
            f"in namespace {namespace}, found {len(pods)}. "
            f"(possibly due to known bug RHAIENG-1819)"
        )
    return pods[0]


@retry(wait_timeout=90, sleep=5)
def wait_for_llama_stack_client_ready(client: LlamaStackClient) -> bool:
    """Wait for LlamaStack client to be ready by checking health, version, and database access."""
    try:
        client.inspect.health()
        version = client.inspect.version()
        models = client.models.list()
        vector_stores = client.vector_stores.list()
        files = client.files.list()
        LOGGER.info(
            f"Llama Stack server is available! "
            f"(version:{version.version} "
            f"models:{len(models)} "
            f"vector_stores:{len(vector_stores.data)} "
            f"files:{len(files.data)})"
        )

    except (APIConnectionError, InternalServerError) as error:
        LOGGER.debug(f"Llama Stack server not ready yet: {error}")
        LOGGER.debug(f"Base URL: {client.base_url}, Error type: {type(error)}, Error details: {error!s}")
        return False

    except Exception as e:  # noqa: BLE001
        LOGGER.warning(f"Unexpected error checking Llama Stack readiness: {e}")
        return False

    else:
        return True
