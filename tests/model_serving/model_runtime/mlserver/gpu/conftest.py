"""GPU-specific fixtures for MLServer CUDA runtime tests.

Provides fixtures for:
- GPU accelerator availability skip logic
- GPU worker node listing
- RBAC fixtures for inference endpoint authorization
- Performance test CPU baseline fixtures
"""

import copy
from collections.abc import Generator
from typing import Any, cast

import pytest
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.node import Node
from ocp_resources.pod import Pod
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.service_account import ServiceAccount
from timeout_sampler import TimeoutSampler

from tests.model_serving.model_runtime.mlserver.constant import PREDICT_RESOURCES
from utilities.constants import (
    KServeDeploymentType,
    ModelFormat,
    ModelInferenceRuntime,
    RuntimeTemplates,
    Timeout,
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


@pytest.fixture(scope="session")
def skip_if_no_gpu_for_mlserver(supported_accelerator_type: str | None) -> None:
    """Fail if no GPU accelerator configured for MLServer GPU tests."""
    if not supported_accelerator_type or supported_accelerator_type.lower() not in {"nvidia"}:
        pytest.fail(f"MLServer GPU tests require nvidia accelerator, got: {supported_accelerator_type}")


@pytest.fixture(scope="session")
def gpu_worker_nodes(admin_client: DynamicClient) -> list[Node]:
    """All cluster nodes with nvidia.com/gpu.present label."""
    return list(Node.get(client=admin_client, label_selector="nvidia.com/gpu.present=true"))


@pytest.fixture(scope="class")
def mlserver_gpu_inference_view_role(
    admin_client: DynamicClient,
    mlserver_inference_service: InferenceService,
) -> Generator[Role]:
    """Creates a Role granting view access to the GPU InferenceService."""
    with create_isvc_view_role(
        client=admin_client,
        isvc=mlserver_inference_service,
        name=f"{mlserver_inference_service.name}-view",
        resource_names=[mlserver_inference_service.name],
    ) as role:
        yield role


@pytest.fixture(scope="class")
def authorized_inference_role_binding(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_model_service_account: ServiceAccount,
    mlserver_gpu_inference_view_role: Role,
) -> Generator[RoleBinding]:
    """Creates a RoleBinding granting the SA inference access."""
    with RoleBinding(
        client=admin_client,
        namespace=model_namespace.name,
        name="mlserver-gpu-inference-rb",
        role_ref_name=mlserver_gpu_inference_view_role.name,
        role_ref_kind=mlserver_gpu_inference_view_role.kind,
        subjects_kind=mlserver_model_service_account.kind,
        subjects_name=mlserver_model_service_account.name,
    ) as rb:
        yield rb


@pytest.fixture(scope="class")
def authorized_inference_token(
    mlserver_model_service_account: ServiceAccount,
    authorized_inference_role_binding: RoleBinding,
) -> str:
    """Returns bearer token for the authorized service account."""
    return create_inference_token(model_service_account=mlserver_model_service_account)


@pytest.fixture(scope="class")
def mlserver_pending_predictor_pods(
    admin_client: DynamicClient,
    mlserver_inference_service: InferenceService,
) -> list[Pod]:
    """Waits up to 2 minutes for predictor pods to appear (may remain Pending)."""
    predictor_pods: list[Pod] = []
    for sample in TimeoutSampler(
        wait_timeout=Timeout.TIMEOUT_2MIN,
        sleep=10,
        exceptions_dict={ResourceNotFoundError: []},
        func=get_pods_by_isvc_label,
        client=admin_client,
        isvc=mlserver_inference_service,
    ):
        if sample:
            predictor_pods = sample
            break
    return predictor_pods


@pytest.fixture(scope="class")
def mlserver_cpu_perf_namespace(admin_client: DynamicClient) -> Generator[Namespace]:
    """Dedicated namespace for CPU baseline ISVC in performance tests."""
    with create_ns(admin_client=admin_client, name="mlserver-perf-cpu") as namespace:
        yield namespace


@pytest.fixture(scope="class")
def mlserver_cpu_perf_isvc(
    admin_client: DynamicClient,
    mlserver_cpu_perf_namespace: Namespace,
    mlserver_runtime_image: str | None,
    s3_models_storage_uri: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
) -> Generator[InferenceService]:
    """CPU MLServer InferenceService in its own namespace for perf baseline."""
    with (
        s3_endpoint_secret(
            client=admin_client,
            name="mlserver-perf-cpu-s3",
            namespace=mlserver_cpu_perf_namespace.name,
            aws_access_key=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_s3_bucket=models_s3_bucket_name,
            aws_s3_region=models_s3_bucket_region,
            aws_s3_endpoint=models_s3_bucket_endpoint,
        ) as cpu_s3_secret,
        ServiceAccount(
            client=admin_client,
            namespace=mlserver_cpu_perf_namespace.name,
            name="mlserver-perf-cpu-sa",
            secrets=[{"name": cpu_s3_secret.name}],
        ) as service_account,
        ServingRuntimeFromTemplate(
            client=admin_client,
            name=ModelInferenceRuntime.MLSERVER_RUNTIME,
            namespace=mlserver_cpu_perf_namespace.name,
            template_name=RuntimeTemplates.MLSERVER,
            deployment_type=KServeDeploymentType.STANDARD,
            runtime_image=mlserver_runtime_image,
        ) as cpu_runtime,
        create_isvc(
            client=admin_client,
            name="resnet-50-onnx",
            namespace=mlserver_cpu_perf_namespace.name,
            runtime=cpu_runtime.name,
            storage_uri=s3_models_storage_uri,
            model_format=ModelFormat.ONNX,
            model_service_account=service_account.name,
            deployment_mode=KServeDeploymentType.STANDARD,
            external_route=True,
            resources=copy.deepcopy(cast(dict[str, Any], PREDICT_RESOURCES["resources"])),
            volumes=copy.deepcopy(PREDICT_RESOURCES["volumes"]),
            volumes_mounts=copy.deepcopy(PREDICT_RESOURCES["volume_mounts"]),
            timeout=Timeout.TIMEOUT_10MIN,
        ) as isvc,
    ):
        yield isvc
