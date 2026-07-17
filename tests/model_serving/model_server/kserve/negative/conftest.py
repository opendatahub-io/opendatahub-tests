from collections.abc import Generator
from typing import Any
from urllib.parse import urlparse

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
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
from utilities.infra import (
    create_inference_token,
    create_isvc_view_role,
    create_ns,
    get_pods_by_isvc_label,
    s3_endpoint_secret,
)
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
        name="neg-kserve",
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
        name="neg-ovms-runtime",
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
        name="neg-ovms-isvc",
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
        name="neg-bad-s3-isvc",
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
    """InferenceService declaring a format not in the runtime's supportedModelFormats."""
    storage_uri = f"s3://{ci_s3_bucket_name}/test-dir/"
    with create_isvc(
        client=admin_client,
        name="neg-wrong-fmt-isvc",
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
        name="neg-corrupted-isvc",
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
        name="neg-bad-pvc-isvc",
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


# --- Auth / RBAC fixtures ---


@pytest.fixture(scope="class")
def auth_service_account(
    unprivileged_client: DynamicClient,
    negative_test_namespace: Namespace,
) -> Generator[ServiceAccount, Any, Any]:
    """ServiceAccount that will be granted inference access via RBAC."""
    with ServiceAccount(
        client=unprivileged_client,
        namespace=negative_test_namespace.name,
        name="neg-auth-sa",
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def negative_test_ovms_isvc_with_auth(
    admin_client: DynamicClient,
    negative_test_namespace: Namespace,
    ovms_serving_runtime: ServingRuntime,
    ci_s3_bucket_name: str,
    negative_test_s3_secret: Secret,
    auth_service_account: ServiceAccount,
) -> Generator[InferenceService, Any, Any]:
    """OVMS InferenceService with auth enabled (kube-rbac-proxy sidecar)."""
    storage_uri = f"s3://{ci_s3_bucket_name}/test-dir/"
    supported_formats = ovms_serving_runtime.instance.spec.supportedModelFormats
    if not supported_formats:
        raise ValueError(f"ServingRuntime '{ovms_serving_runtime.name}' has no supportedModelFormats")

    with create_isvc(
        client=admin_client,
        name="neg-auth-ovms-isvc",
        namespace=negative_test_namespace.name,
        runtime=ovms_serving_runtime.name,
        storage_key=negative_test_s3_secret.name,
        storage_path=urlparse(storage_uri).path,
        model_format=supported_formats[0].name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        model_service_account=auth_service_account.name,
        enable_auth=True,
        external_route=True,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def auth_view_role(
    unprivileged_client: DynamicClient,
    negative_test_ovms_isvc_with_auth: InferenceService,
) -> Generator[Role, Any, Any]:
    """Role granting view access to the auth-protected ISVC."""
    with create_isvc_view_role(
        client=unprivileged_client,
        isvc=negative_test_ovms_isvc_with_auth,
        name=f"{negative_test_ovms_isvc_with_auth.name}-view",
        resource_names=[negative_test_ovms_isvc_with_auth.name],
    ) as role:
        yield role


@pytest.fixture(scope="class")
def auth_role_binding(
    unprivileged_client: DynamicClient,
    auth_view_role: Role,
    auth_service_account: ServiceAccount,
) -> Generator[RoleBinding, Any, Any]:
    """RoleBinding granting the auth SA permission to access the ISVC."""
    with RoleBinding(
        client=unprivileged_client,
        namespace=auth_service_account.namespace,
        name=f"{auth_service_account.name}-view",
        role_ref_name=auth_view_role.name,
        role_ref_kind=auth_view_role.kind,
        subjects_kind=auth_service_account.kind,
        subjects_name=auth_service_account.name,
    ) as rb:
        yield rb


@pytest.fixture(scope="class")
def valid_inference_token(
    auth_service_account: ServiceAccount,
    auth_role_binding: RoleBinding,
) -> str:
    """Valid bearer token that has RBAC access to the auth-protected ISVC."""
    return create_inference_token(model_service_account=auth_service_account)


@pytest.fixture(scope="class")
def cross_namespace_sa_token(
    admin_client: DynamicClient,
) -> Generator[str, Any, Any]:
    """Token from a SA in a different namespace with no access to neg-kserve."""
    with (
        create_ns(
            admin_client=admin_client,
            unprivileged_client=admin_client,
            name="neg-other-ns",
        ) as other_ns,
        ServiceAccount(
            client=admin_client,
            namespace=other_ns.name,
            name="neg-cross-ns-sa",
        ) as sa,
    ):
        token = create_inference_token(model_service_account=sa)
        yield token


@pytest.fixture(scope="class")
def unauthorized_sa_token(
    unprivileged_client: DynamicClient,
    negative_test_namespace: Namespace,
) -> Generator[str, Any, Any]:
    """Token from a SA in the same namespace but with no RBAC for the ISVC."""
    with ServiceAccount(
        client=unprivileged_client,
        namespace=negative_test_namespace.name,
        name="neg-no-rbac-sa",
    ) as sa:
        token = create_inference_token(model_service_account=sa)
        yield token


@pytest.fixture(scope="class")
def initial_pod_state_auth(
    admin_client: DynamicClient,
    negative_test_ovms_isvc_with_auth: InferenceService,
) -> dict[str, dict[str, Any]]:
    """Capture pod state for the auth-protected ISVC."""
    pods = get_pods_by_isvc_label(
        client=admin_client,
        isvc=negative_test_ovms_isvc_with_auth,
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
