from typing import Generator, Any

import pytest
from kubernetes.dynamic import DynamicClient
from pytest_testconfig import config as py_config

from ocp_resources.pod import Pod
from tests.llama_stack.constants import LLS_OPERATOR_POD_FILTER
from utilities.general import wait_for_pods_by_labels


@pytest.fixture(scope="class")
def lls_operator_pod_by_label(admin_client: DynamicClient, model_namespace) -> Generator[Pod, Any, Any]:
    """Get the LLS operator pod."""
    yield wait_for_pods_by_labels(
        admin_client=admin_client,
        namespace=py_config["applications_namespace"],
        label_selector=LLS_OPERATOR_POD_FILTER,
        expected_num_pods=1,
    )[0]
