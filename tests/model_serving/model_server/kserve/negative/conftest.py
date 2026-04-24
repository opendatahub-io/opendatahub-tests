from collections.abc import Generator
from typing import Any
from urllib.parse import urlparse

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime
from pytest_testconfig import config as py_config

from tests.model_serving.model_server.kserve.negative.constants import (
    CORRUPTED_MODEL_S3_PATH,
    INVALID_S3_ACCESS_KEY,
    INVALID_S3_SIGNING_KEY,
    NONEXISTENT_STORAGE_CLASS,
    WRONG_MODEL_FORMAT,
)
from tests.model_serving.model_server.kserve.negative.utils import (
    snapshot_kserve_control_plane_restart_totals,
)
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
def invalid_s3_credentials_secret(
    unprivileged_client: DynamicClient,
    negative_test_namespace: Namespace,
    ci_s3_bucket_name: str,
    ci_s3_bucket_region: str,
    ci_s3_bucket_endpoint: str,
) -> Generator[Secret, Any, Any]:
    """S3 data-connection secret with a valid endpoint and bucket but invalid AWS keys."""
    with s3_endpoint_secret(
        client=unprivileged_client,
        name="invalid-s3-creds-secret",
        namespace=negative_test_namespace.name,
        aws_access_key=INVALID_S3_ACCESS_KEY,
        aws_secret_access_key=INVALID_S3_SIGNING_KEY,
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
def invalid_s3_credentials_ovms_isvc(
    admin_client: DynamicClient,
    negative_test_namespace: Namespace,
    ovms_serving_runtime: ServingRuntime,
    ci_s3_bucket_name: str,
    invalid_s3_credentials_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    """InferenceService that references a real bucket path with intentionally wrong S3 keys."""
    storage_uri = f"s3://{ci_s3_bucket_name}/test-dir/"
    supported_formats = ovms_serving_runtime.instance.spec.supportedModelFormats
    if not supported_formats:
        raise ValueError(f"ServingRuntime '{ovms_serving_runtime.name}' has no supportedModelFormats")

    with create_isvc(
        client=admin_client,
        name="negative-test-invalid-s3-creds-isvc",
        namespace=negative_test_namespace.name,
        runtime=ovms_serving_runtime.name,
        storage_key=invalid_s3_credentials_secret.name,
        storage_path=urlparse(storage_uri).path,
        model_format=supported_formats[0].name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        external_route=False,
        wait=False,
        wait_for_predictor_pods=False,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def wrong_model_format_ovms_isvc(
    admin_client: DynamicClient,
    negative_test_namespace: Namespace,
    ovms_serving_runtime: ServingRuntime,
    ci_s3_bucket_name: str,
    negative_test_s3_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    """InferenceService declaring ``pytorch`` format against an OVMS runtime that expects ONNX."""
    storage_uri = f"s3://{ci_s3_bucket_name}/test-dir/"
    with create_isvc(
        client=admin_client,
        name="negative-test-wrong-format-isvc",
        namespace=negative_test_namespace.name,
        runtime=ovms_serving_runtime.name,
        storage_key=negative_test_s3_secret.name,
        storage_path=urlparse(storage_uri).path,
        model_format=WRONG_MODEL_FORMAT,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        external_route=False,
        wait=False,
        wait_for_predictor_pods=False,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def corrupted_model_ovms_isvc(
    admin_client: DynamicClient,
    negative_test_namespace: Namespace,
    ovms_serving_runtime: ServingRuntime,
    ci_s3_bucket_name: str,
    negative_test_s3_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    """InferenceService pointing to a zero-byte / corrupted model artifact in S3."""
    storage_uri = f"s3://{ci_s3_bucket_name}/{CORRUPTED_MODEL_S3_PATH}/"
    supported_formats = ovms_serving_runtime.instance.spec.supportedModelFormats
    if not supported_formats:
        raise ValueError(f"ServingRuntime '{ovms_serving_runtime.name}' has no supportedModelFormats")

    with create_isvc(
        client=admin_client,
        name="negative-test-corrupted-model-isvc",
        namespace=negative_test_namespace.name,
        runtime=ovms_serving_runtime.name,
        storage_key=negative_test_s3_secret.name,
        storage_path=urlparse(storage_uri).path,
        model_format=supported_formats[0].name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        external_route=False,
        wait=False,
        wait_for_predictor_pods=False,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def bad_storage_class_pvc(
    unprivileged_client: DynamicClient,
    negative_test_namespace: Namespace,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    """PVC referencing a non-existent StorageClass; stays Pending indefinitely."""
    with PersistentVolumeClaim(
        client=unprivileged_client,
        name="bad-sc-pvc",
        namespace=negative_test_namespace.name,
        size="1Gi",
        accessmodes="ReadWriteOnce",
        storage_class=NONEXISTENT_STORAGE_CLASS,
    ) as pvc:
        yield pvc


@pytest.fixture(scope="class")
def bad_pvc_ovms_isvc(
    admin_client: DynamicClient,
    negative_test_namespace: Namespace,
    ovms_serving_runtime: ServingRuntime,
    bad_storage_class_pvc: PersistentVolumeClaim,
) -> Generator[InferenceService, Any, Any]:
    """InferenceService backed by a PVC that cannot be provisioned."""
    supported_formats = ovms_serving_runtime.instance.spec.supportedModelFormats
    if not supported_formats:
        raise ValueError(f"ServingRuntime '{ovms_serving_runtime.name}' has no supportedModelFormats")

    with create_isvc(
        client=admin_client,
        name="negative-test-bad-pvc-isvc",
        namespace=negative_test_namespace.name,
        runtime=ovms_serving_runtime.name,
        storage_uri=f"pvc://{bad_storage_class_pvc.name}/models/",
        model_format=supported_formats[0].name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        external_route=False,
        wait=False,
        wait_for_predictor_pods=False,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def healthy_isvc_pod_restart_baseline(
    admin_client: DynamicClient,
    negative_test_ovms_isvc: InferenceService,
) -> dict[str, int]:
    """Capture per-pod restart totals for the healthy ISVC before a failing neighbor is introduced."""
    pods = get_pods_by_isvc_label(client=admin_client, isvc=negative_test_ovms_isvc)
    if not pods:
        raise AssertionError(
            f"No pods found for InferenceService {negative_test_ovms_isvc.name!r} "
            f"in namespace {negative_test_ovms_isvc.namespace!r} while capturing restart baseline"
        )
    baseline: dict[str, int] = {}
    for pod in pods:
        total = 0
        for cs in pod.instance.status.containerStatuses or []:
            total += cs.restartCount
        for ics in pod.instance.status.initContainerStatuses or []:
            total += ics.restartCount
        baseline[pod.instance.metadata.uid] = total
    return baseline


@pytest.fixture(scope="class")
def kserve_control_plane_restart_baseline(
    admin_client: DynamicClient,
    negative_test_ovms_isvc: InferenceService,
) -> dict[str, int]:
    """Snapshot control-plane restart totals before any failing neighbor ISVC exists."""
    applications_namespace: str = py_config["applications_namespace"]
    return snapshot_kserve_control_plane_restart_totals(
        admin_client=admin_client,
        applications_namespace=applications_namespace,
    )


@pytest.fixture(scope="class")
def neighbor_failing_ovms_isvc(
    admin_client: DynamicClient,
    negative_test_namespace: Namespace,
    ovms_serving_runtime: ServingRuntime,
    ci_s3_bucket_name: str,
    invalid_s3_credentials_secret: Secret,
    negative_test_ovms_isvc: InferenceService,
    healthy_isvc_pod_restart_baseline: dict[str, int],
    kserve_control_plane_restart_baseline: dict[str, int],
) -> Generator[InferenceService, Any, Any]:
    """Failing ISVC deployed alongside a healthy neighbor for isolation testing.

    Depends on ``negative_test_ovms_isvc`` to ensure the healthy ISVC
    is Ready before this failing one is introduced. Pulls in restart baselines
    only for ordering so snapshots run before this ISVC is created.
    """
    _ = (healthy_isvc_pod_restart_baseline, kserve_control_plane_restart_baseline)
    storage_uri = f"s3://{ci_s3_bucket_name}/test-dir/"
    supported_formats = ovms_serving_runtime.instance.spec.supportedModelFormats
    if not supported_formats:
        raise ValueError(f"ServingRuntime '{ovms_serving_runtime.name}' has no supportedModelFormats")

    with create_isvc(
        client=admin_client,
        name="neg-fail-neighbor-isvc",
        namespace=negative_test_namespace.name,
        runtime=ovms_serving_runtime.name,
        storage_key=invalid_s3_credentials_secret.name,
        storage_path=urlparse(storage_uri).path,
        model_format=supported_formats[0].name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        external_route=False,
        wait=False,
        wait_for_predictor_pods=False,
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
