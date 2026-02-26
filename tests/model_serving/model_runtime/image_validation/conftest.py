"""
Fixtures for serving runtime image validation tests.

Creates minimal ServingRuntime + InferenceService so that deployments/pods
are created and their spec.containers[*].image can be validated against
the CSV relatedImages (registry.redhat.io, sha256 digest).
"""

from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from timeout_sampler import TimeoutSampler

from tests.model_serving.model_runtime.image_validation.constant import PLACEHOLDER_STORAGE_URI
from utilities.constants import KServeDeploymentType
from utilities.inference_utils import create_isvc
from utilities.infra import create_ns, wait_for_isvc_pods
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="class")
def serving_runtime_image_validation_namespace(
    admin_client: DynamicClient,
) -> Generator[Namespace, Any, Any]:
    """
    A dedicated namespace for serving runtime image validation.

    Ensures deployments/pods created by the test have a clean namespace
    that is torn down after the test.
    """
    name = "runtime-verification"
    with create_ns(admin_client=admin_client, name=name, teardown=True) as ns:
        yield ns


@pytest.fixture(scope="function")
def serving_runtime_pods_for_runtime(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    serving_runtime_image_validation_namespace: Namespace,
) -> Generator[tuple[list[Pod], str], Any, Any]:
    """
    For a given runtime config (parametrized), create ServingRuntime + InferenceService,
    wait for pods, yield (pods, display_name) for validation. Teardown after test.
    """
    config = request.param
    display_name = config["name"]
    name_slug = display_name.replace("_", "-")
    namespace_name = serving_runtime_image_validation_namespace.name
    runtime_name = f"{name_slug}-runtime"
    isvc_name = f"{name_slug}-isvc"

    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=runtime_name,
        namespace=namespace_name,
        template_name=config["template"],
        deployment_type="raw",
    ) as serving_runtime:
        # Get model format from the runtime for the InferenceService spec.
        model_format = serving_runtime.instance.spec.supportedModelFormats[0].name
        with create_isvc(
            client=admin_client,
            name=isvc_name,
            namespace=namespace_name,
            model_format=model_format,
            runtime=runtime_name,
            storage_uri=PLACEHOLDER_STORAGE_URI,
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            wait=False,
            wait_for_predictor_pods=False,
            timeout=120,
            teardown=True,
        ) as isvc:
            # Wait for pods to be created (300 seconds timeout)
            for pods in TimeoutSampler(
                wait_timeout=300,
                sleep=5,
                func=wait_for_isvc_pods,
                client=admin_client,
                isvc=isvc,
                runtime_name=runtime_name,
            ):
                if pods:
                    yield (pods, display_name)
                    return
