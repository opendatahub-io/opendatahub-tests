from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod

from tests.llama_stack.constants import LLS_CORE_POD_FILTER
from utilities.general import wait_for_pods_by_labels


@pytest.fixture(scope="class")
def llama_stack_distribution_pods(
    admin_client: DynamicClient, unprivileged_model_namespace: Namespace
) -> Generator[Pod, Any, Any]:
    """Returns the LlamaStackDistribution pods running in the namespace."""
    yield wait_for_pods_by_labels(
        admin_client=admin_client,
        namespace=unprivileged_model_namespace.name,
        label_selector=LLS_CORE_POD_FILTER,
        expected_num_pods=1,
    )[0]
