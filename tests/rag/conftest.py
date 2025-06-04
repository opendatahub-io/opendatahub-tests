from typing import Dict, Generator, Any
import pytest
import os
import shlex
import shortuuid
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from _pytest.fixtures import FixtureRequest
from ocp_resources.namespace import Namespace
from utilities.infra import create_ns
from pyhelper_utils.shell import run_command
from simple_logger.logger import get_logger
from utilities.rag_utils import create_llama_stack_distribution, LlamaStackDistribution

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
                {"name": "MILVUS_DB_PATH", "value": '/.llama/distributions/remote-vllm/milvus.db'}
            ],
            "name": "llama-stack",
            "port": 8321
        },
        "distribution": {
            "image": 'quay.io/mcampbel/llama-stack:milvus-granite-embedding-125m-english'
        },
        "podOverrides": {
            "volumeMounts": [
                {"mountPath": "/root/.llama", "name": "llama-storage"}
            ],
            "volumes": [
                {"emptyDir": {}, "name": "llama-storage"}
            ]
        }
    }

@pytest.fixture(scope="session")
def installed_llama_stack_operator(admin_client: DynamicClient) -> Generator[None, Any, Any]:
    LLAMA_STACK_REPO = "https://raw.githubusercontent.com/llamastack/llama-stack-k8s-operator"
    LLAMA_STACK_OPERATOR_DEPLOY_YAML = LLAMA_STACK_REPO + "/main/release/operator.yaml"
    operator_ns = Namespace(name="llama-stack-k8s-operator-system")
    operator_name = "llama-stack-k8s-operator-controller-manager"

    deployment = Deployment(
            client=admin_client,
            namespace=operator_ns.name,
            name=f"{operator_name}",
        )
    if not deployment.exists:
        cmd = f"oc apply -f {LLAMA_STACK_OPERATOR_DEPLOY_YAML}"
        LOGGER.debug(f"Executing command: {cmd}")
        run_command(command=shlex.split(cmd), verify_stderr=False, check=True, timeout=30)
        
    deployment.wait_for_replicas()

    yield
    cmd = f"oc delete -f {LLAMA_STACK_OPERATOR_DEPLOY_YAML}"
    LOGGER.debug(f"Executing command: {cmd}")
    run_command(command=shlex.split(cmd), verify_stderr=False, check=True, timeout=30)

@pytest.fixture(scope="function")
def rag_test_namespace(unprivileged_client: DynamicClient) -> Generator[Namespace, Any, Any]:
    namespace_name = "rag-test-" + shortuuid.uuid().lower()
    with create_ns(namespace_name, unprivileged_client=unprivileged_client) as ns:
        yield ns

@pytest.fixture(scope="function")
def llama_stack_distribution_from_template(
    installed_llama_stack_operator: Generator[None, Any, Any],
    rag_test_namespace: Namespace,
    request: FixtureRequest,
    admin_client: DynamicClient) -> Generator[LlamaStackDistribution, Any, Any]:
    with create_llama_stack_distribution(
        client=admin_client,
        name="rag-llama-stack-distribution",
        namespace=rag_test_namespace.name,
        replicas=1,
        server=llama_stack_server()
    ) as llama_stack_distribution:
        yield llama_stack_distribution

@pytest.fixture(scope="function")
def llama_stack_distribution_deployment(
    rag_test_namespace: Namespace,
    admin_client: DynamicClient,
    llama_stack_distribution_from_template: Generator[LlamaStackDistribution, Any, Any]
) -> Generator[Deployment, Any, Any]:
    deployment = Deployment(
            client=admin_client,
            namespace=rag_test_namespace.name,
            name="rag-llama-stack-distribution",
        )
    assert deployment.exists
    yield deployment
