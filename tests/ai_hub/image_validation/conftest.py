from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from pytest import FixtureRequest

from tests.ai_hub.utils import get_latest_job_pod
from utilities.general import wait_for_pods_by_labels
from utilities.openshift_resources.job import Job
from utilities.openshift_resources.pod import Pod


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
def resource_pods(request: FixtureRequest) -> list[Pod]:
    namespace = request.param.get("namespace")
    label_selector = request.param.get("label_selector")
    assert namespace
    return list(Pod.list_resources(namespace=namespace, label_selector=label_selector))


@pytest.fixture(scope="class")
def async_job_pod(
    async_upload_image: str,
    model_registry_namespace: str,
) -> Generator[Pod, Any, Any]:
    """Deploy async upload job with a trivial command and yield the resulting pod."""
    with Job(
        name="async-upload-image-validation",
        namespace=model_registry_namespace,
        template={
            "spec": {
                "restartPolicy": "Never",
                "containers": [
                    {
                        "name": "async-upload",
                        "image": async_upload_image,
                        "command": ["true"],
                    }
                ],
            }
        },
    ) as job:
        job.wait_for_condition(condition="Complete", status="True")
        yield get_latest_job_pod(job=job)
