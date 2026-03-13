from collections.abc import Generator
from typing import Any
from urllib.parse import urlparse

import pytest
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
from utilities.infra import create_ns, get_pods_by_isvc_label, s3_endpoint_secret
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="package")
def negative_test_namespace(
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
) -> Generator[Namespace, Any, Any]:
    """Create a shared namespace for all negative tests."""
    with create_ns(
        admin_client=admin_client,
        unprivileged_client=unprivileged_client,
        name="negative-test-kserve",
    ) as ns:
        yield ns


@pytest.fixture(scope="package")
def negative_test_s3_secret(
    unprivileged_client: DynamicClient,
    negative_test_namespace: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    ci_s3_bucket_name: str,
    ci_s3_bucket_region: str,
    ci_s3_bucket_endpoint: str,
) -> Generator[Secret, Any, Any]:
    """Create S3 secret shared across all negative tests."""
    with s3_endpoint_secret(
        client=unprivileged_client,
        name="ci-bucket-secret",
        namespace=negative_test_namespace.name,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_region=ci_s3_bucket_region,
        aws_s3_bucket=ci_s3_bucket_name,
        aws_s3_endpoint=ci_s3_bucket_endpoint,
    ) as secret:
        yield secret


@pytest.fixture(scope="package")
def ovms_serving_runtime(
    admin_client: DynamicClient,
    negative_test_namespace: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    """Create OVMS serving runtime shared across all negative tests."""
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="negative-test-ovms-runtime",
        namespace=negative_test_namespace.name,
        template_name=RuntimeTemplates.OVMS_KSERVE,
        multi_model=False,
        enable_http=True,
        enable_grpc=False,
    ) as runtime:
        yield runtime


@pytest.fixture(scope="package")
def negative_test_ovms_isvc(
    admin_client: DynamicClient,
    negative_test_namespace: Namespace,
    ovms_serving_runtime: ServingRuntime,
    ci_s3_bucket_name: str,
    negative_test_s3_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    """Create InferenceService with OVMS runtime shared across all negative tests."""
    storage_uri = f"s3://{ci_s3_bucket_name}/test-dir/"
    supported_formats = ovms_serving_runtime.instance.spec.supportedModelFormats
    if not supported_formats:
        raise ValueError(f"ServingRuntime '{ovms_serving_runtime.name}' has no supportedModelFormats")

    with create_isvc(
        client=admin_client,
        name="negative-test-ovms-isvc",
        namespace=negative_test_namespace.name,
        runtime=ovms_serving_runtime.name,
        storage_key=negative_test_s3_secret.name,
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
