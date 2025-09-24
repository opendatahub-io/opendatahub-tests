from typing import Generator, Any

import pytest
from kubernetes.dynamic import DynamicClient

from ocp_resources.pod import Pod
from tests.llama_stack.constants import LLS_CORE_POD_FILTER
from utilities.general import wait_for_pods_by_labels


@pytest.fixture(scope="class")
def lls_pods(admin_client: DynamicClient, model_namespace) -> Generator[Pod, Any, Any]:
    """Get the LLS core deployment pod."""
    yield wait_for_pods_by_labels(
        admin_client=admin_client,
        namespace=model_namespace.name,
        label_selector=LLS_CORE_POD_FILTER,
        expected_num_pods=1,
    )[0]
