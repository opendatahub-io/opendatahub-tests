from typing import Generator, Any

import pytest
from kubernetes.dynamic import DynamicClient
from pytest_testconfig import config as py_config

from ocp_resources.pod import Pod
from tests.model_registry.constants import MODEL_REGISTRY_POD_FILTER
from utilities.general import wait_for_pods_by_labels


@pytest.fixture(scope="function")
def model_registry_instance_pod_by_label(admin_client: DynamicClient) -> Generator[Pod, Any, Any]:
    """Get the model registry instance pod."""
    yield wait_for_pods_by_labels(
        admin_client=admin_client,
        namespace=py_config["model_registry_namespace"],
        label_selector=MODEL_REGISTRY_POD_FILTER,
        expected_num_pods=1,
    )[0]
