from typing import Generator, Any

import pytest
import shlex
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from pyhelper_utils.shell import run_command
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="session")
def installed_llama_stack_operator(admin_client: DynamicClient) -> Generator[None, Any, Any]:
    LLAMA_STACK_REPO = "https://raw.githubusercontent.com/llamastack/llama-stack-k8s-operator"
    LLAMA_STACK_OPERATOR_DEPLOY_YAML = LLAMA_STACK_REPO + "/main/release/operator.yaml"
    operator_ns = Namespace(name="llama-stack-k8s-operator-system", ensure_exists=False)
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
