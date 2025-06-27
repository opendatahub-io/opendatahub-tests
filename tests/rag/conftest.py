from typing import Dict, Generator, Any
import pytest
import os
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from _pytest.fixtures import FixtureRequest
from ocp_resources.namespace import Namespace
from utilities.infra import create_ns
from simple_logger.logger import get_logger
from utilities.rag_utils import create_llama_stack_distribution, LlamaStackDistribution
from utilities.data_science_cluster_utils import update_components_in_dsc
from utilities.constants import DscComponents, Timeout
from utilities.general import generate_random_name
from timeout_sampler import TimeoutSampler

LOGGER = get_logger(name=__name__)


def llama_stack_server() -> Dict[str, Any]:
    rag_vllm_url = os.getenv("RAG_VLLM_URL")
    rag_vllm_model = os.getenv("RAG_VLLM_MODEL")
    rag_vllm_token = os.getenv("RAG_VLLM_TOKEN")

    return {
        "containerSpec": {
            "env": [
                {"name": "INFERENCE_MODEL", "value": rag_vllm_model},
                {"name": "VLLM_TLS_VERIFY", "value": "false"},
                {"name": "VLLM_API_TOKEN", "value": rag_vllm_token},
                {"name": "VLLM_URL", "value": rag_vllm_url},
                {"name": "MILVUS_DB_PATH", "value": "/.llama/distributions/remote-vllm/milvus.db"},
            ],
            "name": "llama-stack",
            "port": 8321,
        },
        "distribution": {"image": "quay.io/mcampbel/llama-stack:milvus-granite-embedding-125m-english"},
        "podOverrides": {
            "volumeMounts": [{"mountPath": "/root/.llama", "name": "llama-storage"}],
            "volumes": [{"emptyDir": {}, "name": "llama-storage"}],
        },
    }


@pytest.fixture(scope="class")
def enabled_llama_stack_operator(dsc_resource: DataScienceCluster) -> Generator[None, Any, Any]:
    with update_components_in_dsc(
        dsc=dsc_resource,
        components={
            DscComponents.LLAMASTACKOPERATOR: DscComponents.ManagementState.MANAGED,
        },
        wait_for_components_state=True,
    ) as dsc:
        yield dsc


@pytest.fixture(scope="function")
def rag_test_namespace(unprivileged_client: DynamicClient) -> Generator[Namespace, Any, Any]:
    namespace_name = generate_random_name(prefix="rag-test-")
    with create_ns(namespace_name, unprivileged_client=unprivileged_client) as ns:
        yield ns


@pytest.fixture(scope="function")
def llama_stack_distribution_from_template(
    enabled_llama_stack_operator: Generator[None, Any, Any],
    rag_test_namespace: Namespace,
    request: FixtureRequest,
    admin_client: DynamicClient,
) -> Generator[LlamaStackDistribution, Any, Any]:
    with create_llama_stack_distribution(
        client=admin_client,
        name="rag-llama-stack-distribution",
        namespace=rag_test_namespace.name,
        replicas=1,
        server=llama_stack_server(),
    ) as llama_stack_distribution:
        yield llama_stack_distribution


@pytest.fixture(scope="function")
def llama_stack_distribution_deployment(
    rag_test_namespace: Namespace,
    admin_client: DynamicClient,
    llama_stack_distribution_from_template: Generator[LlamaStackDistribution, Any, Any],
) -> Generator[Deployment, Any, Any]:
    deployment = Deployment(
        client=admin_client,
        namespace=rag_test_namespace.name,
        name="rag-llama-stack-distribution",
    )

    timeout = Timeout.TIMEOUT_15_SEC
    sampler = TimeoutSampler(
        wait_timeout=timeout, sleep=1, func=lambda deployment: deployment.exists is not None, deployment=deployment
    )
    for item in sampler:
        if item:
            break  # Break after first successful iteration

    assert deployment.exists, f"llama stack distribution deployment doesn't exist within {timeout} seconds"
    yield deployment
