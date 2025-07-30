import os
from typing import Any, Dict, Generator

import portforward
import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from llama_stack_client import LlamaStackClient, APIConnectionError
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.project_project_openshift_io import Project
from simple_logger.logger import get_logger
from timeout_sampler import retry

from utilities.constants import DscComponents, Timeout
from utilities.data_science_cluster_utils import update_components_in_dsc
from utilities.general import generate_random_name
from utilities.infra import create_ns
from ocp_resources.llama_stack_distribution import LlamaStackDistribution
from utilities.rag_utils import create_llama_stack_distribution

LOGGER = get_logger(name=__name__)


def llama_stack_server() -> Dict[str, Any]:
    rag_vllm_url = os.getenv("RAG_VLLM_URL")
    rag_vllm_model = os.getenv("RAG_VLLM_MODEL")
    rag_vllm_token = os.getenv("RAG_VLLM_TOKEN")

    return {
        "containerSpec": {
            "resources": {
                "requests": {"cpu": "250m", "memory": "500Mi"},
                "limits": {"cpu": "2", "memory": "12Gi"},
            },
            "env": [
                {"name": "INFERENCE_MODEL", "value": rag_vllm_model},
                {"name": "VLLM_TLS_VERIFY", "value": "false"},
                {"name": "VLLM_API_TOKEN", "value": rag_vllm_token},
                {"name": "VLLM_URL", "value": rag_vllm_url},
                {"name": "FMS_ORCHESTRATOR_URL", "value": "http://localhost"},
            ],
            "name": "llama-stack",
            "port": 8321,
        },
        "distribution": {"name": "rh-dev"},
    }


@pytest.fixture(scope="class")
def enabled_llama_stack_operator(dsc_resource: DataScienceCluster) -> Generator[DataScienceCluster, Any, Any]:
    with update_components_in_dsc(
        dsc=dsc_resource,
        components={
            DscComponents.LLAMASTACKOPERATOR: DscComponents.ManagementState.MANAGED,
        },
        wait_for_components_state=True,
    ) as dsc:
        yield dsc


@pytest.fixture(scope="class")
def rag_test_namespace(
    admin_client: DynamicClient, unprivileged_client: DynamicClient
) -> Generator[Namespace | Project, Any, Any]:
    namespace_name = generate_random_name(prefix="rag-test")
    with create_ns(name=namespace_name, admin_client=admin_client, unprivileged_client=unprivileged_client) as ns:
        yield ns


@pytest.fixture(scope="class")
def llama_stack_distribution_from_template(
    enabled_llama_stack_operator: Generator[DataScienceCluster, Any, Any],
    rag_test_namespace: Namespace | Project,
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


@pytest.fixture(scope="class")
def llama_stack_distribution_deployment(
    rag_test_namespace: Namespace | Project,
    admin_client: DynamicClient,
    llama_stack_distribution_from_template: Generator[LlamaStackDistribution, Any, Any],
) -> Generator[Deployment, Any, Any]:
    deployment = Deployment(
        client=admin_client,
        namespace=rag_test_namespace.name,
        name="rag-llama-stack-distribution",
    )

    deployment.wait(timeout=Timeout.TIMEOUT_2MIN)
    yield deployment


@retry(wait_timeout=Timeout.TIMEOUT_1MIN, sleep=5)
def wait_for_llama_stack_ready(client: LlamaStackClient) -> bool:
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


@pytest.fixture(scope="class")
def rag_lls_client(
    admin_client: DynamicClient,
    rag_test_namespace: Namespace | Project,
    llama_stack_distribution_deployment: Deployment,
) -> Generator[LlamaStackClient, Any, Any]:
    """
    Returns a ready to use LlamaStackClient,  enabling port forwarding
    from the llama-stack-server service:8321 to localhost:8321

    Args:
        admin_client (DynamicClient): Kubernetes dynamic client for cluster operations
        rag_test_namespace (Namespace | Project): Namespace or project containing RAG test resources
        llama_stack_distribution_deployment (Deployment): LlamaStack distribution deployment resource

    Yields:
        Generator[LlamaStackClient, Any, Any]: Configured LlamaStackClient for RAG testing
    """
    try:
        with portforward.forward(
            pod_or_service="rag-llama-stack-distribution-service",
            namespace=rag_test_namespace.name,
            from_port=8321,
            to_port=8321,
            waiting=15,
        ):
            client = LlamaStackClient(
                base_url="http://localhost:8321",
                timeout=120.0,
            )
            wait_for_llama_stack_ready(client=client)
            yield client
    except Exception as e:
        LOGGER.error(f"Failed to set up port forwarding: {e}")
        raise
