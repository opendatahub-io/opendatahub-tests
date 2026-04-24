from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.pod import Pod
from pytest import FixtureRequest
from pytest_testconfig import config as py_config

from utilities.general import wait_for_pods_by_labels
from utilities.infra import ResourceNotFoundError


@pytest.fixture(scope="class")
def model_registry_instance_pods_by_label(
    request: FixtureRequest, admin_client: DynamicClient, model_registry_namespace: str
) -> Generator[list[Pod], Any, Any]:
    """Get the model registry instance pod."""
    pods = [
        wait_for_pods_by_labels(
            admin_client=admin_client,
            namespace=model_registry_namespace,
            label_selector=label,
            expected_num_pods=1,
        )[0]
        for label in request.param["label_selectors"]
    ]
    yield pods


@pytest.fixture(scope="function")
def resource_pods(request: FixtureRequest, admin_client: DynamicClient) -> list[Pod]:
    namespace = request.param.get("namespace")
    label_selector = request.param.get("label_selector")
    assert namespace
    return list(Pod.get(namespace=namespace, label_selector=label_selector, client=admin_client))


@pytest.fixture(scope="session")
def async_upload_image(admin_client: DynamicClient) -> str:
    """Async upload job image from the model-registry-operator-parameters ConfigMap."""
    config_map = ConfigMap(
        client=admin_client,
        name="model-registry-operator-parameters",
        namespace=py_config["applications_namespace"],
    )

    if not config_map.exists:
        raise ResourceNotFoundError(
            f"ConfigMap 'model-registry-operator-parameters' not found in"
            f" namespace '{py_config['applications_namespace']}'"
        )

    return config_map.instance.data["IMAGES_JOBS_ASYNC_UPLOAD"]
