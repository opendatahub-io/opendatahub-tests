"""
Pytest fixtures for MLServer model serving runtime tests.

This module provides fixtures for:
- Setting up MLServer serving runtimes using pre-installed templates
- Creating inference services and related Kubernetes resources
- Managing S3 secrets and service accounts
- Providing test utilities like snapshots and pod resources
"""

import copy
from collections.abc import Generator
from typing import Any, cast

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from syrupy.extensions.json import JSONSnapshotExtension

from tests.model_serving.model_runtime.mlserver.constant import (
    MLSERVER_ACCELERATOR_IDENTIFIER,
    MLSERVER_RUNTIME_NAME_MAP,
    MLSERVER_TEMPLATE_MAP,
    PREDICT_RESOURCES,
)
from utilities.constants import (
    KServeDeploymentType,
    Labels,
    ModelInferenceRuntime,
    RuntimeTemplates,
)
from utilities.inference_utils import create_isvc
from utilities.infra import get_pods_by_isvc_label
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="class")
def mlserver_serving_runtime(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    supported_accelerator_type: str | None,
    mlserver_runtime_image: str | None,
) -> Generator[ServingRuntime]:
    """Provides a ServingRuntime, selecting CPU or CUDA template based on params.

    Uses CUDA template only when request.param contains 'gpu': True.
    Otherwise uses the CPU template regardless of --supported-accelerator-type.

    Args:
        request: Pytest fixture request containing deployment parameters.
        admin_client: Kubernetes dynamic client.
        model_namespace: Kubernetes namespace for model deployment.
        supported_accelerator_type: Accelerator type from CLI (e.g., 'nvidia').
        mlserver_runtime_image: Optional container image override for the runtime.

    Yields:
        ServingRuntime: An instance of the MLServer ServingRuntime (CPU or CUDA).
    """
    params = request.param if hasattr(request, "param") and isinstance(request.param, dict) else {}
    use_gpu = params.get("gpu", False)

    if use_gpu:
        accelerator = (supported_accelerator_type or "").lower()
        template_name = MLSERVER_TEMPLATE_MAP.get(accelerator, RuntimeTemplates.MLSERVER_CUDA)
        runtime_name = MLSERVER_RUNTIME_NAME_MAP.get(accelerator, ModelInferenceRuntime.MLSERVER_CUDA_RUNTIME)
    else:
        template_name = RuntimeTemplates.MLSERVER
        runtime_name = ModelInferenceRuntime.MLSERVER_RUNTIME

    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=runtime_name,
        namespace=model_namespace.name,
        template_name=template_name,
        deployment_type=request.param["deployment_mode"],
        runtime_image=mlserver_runtime_image,
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def mlserver_inference_service(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_serving_runtime: ServingRuntime,
    supported_accelerator_type: str | None,
    s3_models_storage_uri: str,
    mlserver_model_service_account: ServiceAccount,
) -> Generator[InferenceService]:
    """Creates a configured InferenceService with dynamic accelerator-aware GPU resources.

    Supports optional GPU allocation via gpu_count param. When gpu_count > 0 and
    supported_accelerator_type is set, the appropriate GPU identifier and resource
    limits are applied to the ISVC container spec.

    Args:
        request: Pytest fixture request containing test parameters.
        admin_client: Kubernetes dynamic client.
        model_namespace: Kubernetes namespace for model deployment.
        mlserver_serving_runtime: The MLServer ServingRuntime instance.
        supported_accelerator_type: Accelerator type from CLI (e.g., 'nvidia').
        s3_models_storage_uri: URI for the S3 storage location of models.
        mlserver_model_service_account: Service account for the model.

    Yields:
        InferenceService: A configured InferenceService resource.
    """
    params = request.param
    service_config = {
        "client": admin_client,
        "name": params.get("name"),
        "namespace": model_namespace.name,
        "runtime": mlserver_serving_runtime.name,
        "storage_uri": s3_models_storage_uri,
        "model_format": params.get(
            "model_format", mlserver_serving_runtime.instance.spec.supportedModelFormats[0].name
        ),
        "model_service_account": mlserver_model_service_account.name,
        "deployment_mode": params.get("deployment_mode", KServeDeploymentType.STANDARD),
        "external_route": params.get("enable_external_route", False),
        "enable_auth": params.get("enable_auth", False),
    }

    gpu_count = params.get("gpu_count", 0)
    timeout = params.get("timeout")
    min_replicas = params.get("min-replicas")

    resources = copy.deepcopy(cast(dict[str, dict[str, str]], PREDICT_RESOURCES["resources"]))
    if gpu_count > 0:
        accelerator = (supported_accelerator_type or "").lower()
        identifier = MLSERVER_ACCELERATOR_IDENTIFIER.get(accelerator, Labels.Nvidia.NVIDIA_COM_GPU)
        resources["requests"][identifier] = gpu_count
        resources["limits"][identifier] = gpu_count

    service_config["resources"] = resources
    service_config["volumes"] = copy.deepcopy(PREDICT_RESOURCES["volumes"])
    service_config["volumes_mounts"] = copy.deepcopy(PREDICT_RESOURCES["volume_mounts"])

    if timeout:
        service_config["timeout"] = timeout

    if min_replicas is not None:
        service_config["min_replicas"] = min_replicas

    service_config["wait"] = params.get("wait", True)
    service_config["wait_for_predictor_pods"] = params.get("wait_for_predictor_pods", True)

    with create_isvc(**service_config) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def mlserver_model_service_account(admin_client: DynamicClient, kserve_s3_secret: Secret) -> Generator[ServiceAccount]:
    """
    Creates and yields a ServiceAccount linked to the provided S3 secret for MLServer models.

    Args:
        admin_client (DynamicClient): Kubernetes dynamic client.
        kserve_s3_secret (Secret): The Kubernetes secret containing S3 credentials.

    Yields:
        ServiceAccount: A ServiceAccount configured with access to the S3 secret.
    """
    with ServiceAccount(
        client=admin_client,
        namespace=kserve_s3_secret.namespace,
        name="mlserver-models-bucket-sa",
        secrets=[{"name": kserve_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def mlserver_model_car_inference_service(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_serving_runtime: ServingRuntime,
) -> Generator[InferenceService]:
    """
    Create InferenceService for MLServer model car (OCI image) testing.

    Args:
        request: Pytest fixture request with parameters.
        admin_client: Kubernetes dynamic client.
        model_namespace: Namespace for deployment.
        mlserver_serving_runtime: MLServer ServingRuntime instance.

    Yields:
        InferenceService: Configured ISVC using OCI storage.
    """
    params = request.param
    storage_uri = params.get("storage-uri")
    if not storage_uri:
        raise ValueError("storage-uri is required in params")

    deployment_mode = params.get("deployment_mode", KServeDeploymentType.STANDARD)
    model_format = params.get("model-format")
    if not model_format:
        raise ValueError("model-format is required in params")

    with create_isvc(
        client=admin_client,
        name=f"{model_format}-modelcar",
        namespace=model_namespace.name,
        runtime=mlserver_serving_runtime.name,
        storage_uri=storage_uri,
        model_format=model_format,
        deployment_mode=deployment_mode,
        external_route=params.get("enable_external_route", False),
        wait_for_predictor_pods=params.get("wait_for_predictor_pods", False),
        model_env_variables=params.get("model_env_variables"),
    ) as isvc:
        yield isvc


@pytest.fixture
def mlserver_response_snapshot(snapshot: Any) -> Any:
    """
    Provides a snapshot fixture configured to use JSONSnapshotExtension for MLServer responses.

    Args:
        snapshot (Any): The base snapshot fixture.

    Returns:
        Any: Snapshot fixture extended with JSONSnapshotExtension.
    """
    return snapshot.use_extension(extension_class=JSONSnapshotExtension)


@pytest.fixture
def mlserver_pod_resource(
    admin_client: DynamicClient,
    mlserver_inference_service: InferenceService,
) -> Pod:
    """
    Retrieves the first Kubernetes Pod associated with the given MLServer InferenceService.

    Args:
        admin_client (DynamicClient): Kubernetes dynamic client.
        mlserver_inference_service (InferenceService): The MLServer InferenceService resource.

    Returns:
        Pod: The first Pod found for the InferenceService.

    Raises:
        RuntimeError: If no pods are found for the specified InferenceService.
    """
    pods = get_pods_by_isvc_label(client=admin_client, isvc=mlserver_inference_service)
    if not pods:
        raise RuntimeError(f"No pods found for InferenceService {mlserver_inference_service.name}")
    return pods[0]
