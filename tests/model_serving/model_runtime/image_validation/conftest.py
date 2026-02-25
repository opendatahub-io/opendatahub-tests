"""
Fixtures for serving runtime image validation tests.

Creates minimal ServingRuntime + InferenceService so that deployments/pods
are created and their spec.containers[*].image can be validated against
the CSV relatedImages (registry.redhat.io, sha256 digest).
"""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from timeout_sampler import TimeoutSampler

from tests.model_serving.model_runtime.image_validation.constant import (
    PLACEHOLDER_STORAGE_URI,
    POD_WAIT_TIMEOUT,
)
from utilities.constants import KServeDeploymentType, Timeout
from utilities.inference_utils import create_isvc
from utilities.infra import create_ns, get_pods_by_isvc_label
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

    with _serving_runtime_and_isvc(
        admin_client=admin_client,
        namespace_name=namespace_name,
        runtime_name=runtime_name,
        isvc_name=isvc_name,
        template_name=config["template"],
    ) as isvc:
        pods = _wait_for_isvc_pods(
            client=admin_client,
            isvc=isvc,
            runtime_name=runtime_name,
            timeout=POD_WAIT_TIMEOUT,
        )
        yield (pods, display_name)


@contextmanager
def _serving_runtime_and_isvc(
    admin_client: DynamicClient,
    namespace_name: str,
    runtime_name: str,
    isvc_name: str,
    template_name: str,
) -> Generator[InferenceService]:
    """Create ServingRuntime from template and minimal InferenceService; yield ISVC."""
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=runtime_name,
        namespace=namespace_name,
        template_name=template_name,
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
            timeout=Timeout.TIMEOUT_2MIN,
            teardown=True,
        ) as isvc:
            yield isvc


def _wait_for_isvc_pods(
    client: DynamicClient,
    isvc: InferenceService,
    runtime_name: str,
    timeout: int,
) -> list[Pod]:
    """Wait until at least one pod exists for the InferenceService; return list of pods."""

    def _get_pods() -> list[Pod] | None:
        try:
            return get_pods_by_isvc_label(
                client=client,
                isvc=isvc,
                runtime_name=runtime_name,
            )
        except ResourceNotFoundError:
            return None

    for pods in TimeoutSampler(
        wait_timeout=timeout,
        sleep=5,
        func=_get_pods,
    ):
        if pods:
            return pods
    raise TimeoutError(
        f"No pods found for InferenceService {isvc.name} in namespace {isvc.namespace} within {timeout}s"
    )
