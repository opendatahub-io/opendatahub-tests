from typing import Any, Generator

from urllib.parse import urlparse

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime

from utilities.constants import (
    KServeDeploymentType,
    RuntimeTemplates,
)
from utilities.inference_utils import create_isvc
from utilities.infra import get_pods_by_isvc_label
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="class")
def ovms_serving_runtime(
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    """Create OVMS serving runtime for negative tests."""
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="negative-test-ovms-runtime",
        namespace=unprivileged_model_namespace.name,
        template_name=RuntimeTemplates.OVMS_KSERVE,
        multi_model=False,
        enable_http=True,
        enable_grpc=False,
    ) as runtime:
        yield runtime


@pytest.fixture(scope="class")
def negative_test_ovms_isvc(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    ovms_serving_runtime: ServingRuntime,
    ci_s3_bucket_name: str,
    ci_endpoint_s3_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    """Create InferenceService with OVMS runtime for negative tests."""
    storage_uri = f"s3://{ci_s3_bucket_name}/{request.param['model-dir']}/"
    supported_formats = ovms_serving_runtime.instance.spec.supportedModelFormats
    if not supported_formats:
        raise ValueError(f"ServingRuntime '{ovms_serving_runtime.name}' has no supportedModelFormats")

    with create_isvc(
        client=admin_client,
        name="negative-test-ovms-isvc",
        namespace=unprivileged_model_namespace.name,
        runtime=ovms_serving_runtime.name,
        storage_key=ci_endpoint_s3_secret.name,
        storage_path=urlparse(storage_uri).path,
        model_format=supported_formats[0].name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        external_route=True,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def initial_pod_state(
    admin_client: DynamicClient,
    negative_test_ovms_isvc: InferenceService,
) -> dict[str, dict[str, Any]]:
    """Capture initial pod state (UIDs, restart counts) before tests run.

    Returns:
        A dictionary mapping pod UIDs to their initial state including
        name, restart counts per container.
    """
    pods = get_pods_by_isvc_label(
        client=admin_client,
        isvc=negative_test_ovms_isvc,
    )

    pod_state: dict[str, dict[str, Any]] = {}
    for pod in pods:
        uid = pod.instance.metadata.uid
        container_restart_counts = {
            container.name: container.restartCount for container in (pod.instance.status.containerStatuses or [])
        }
        pod_state[uid] = {
            "name": pod.name,
            "restart_counts": container_restart_counts,
        }

    return pod_state
