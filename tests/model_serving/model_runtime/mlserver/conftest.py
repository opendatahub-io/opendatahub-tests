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
    mlserver_runtime_image: str,
) -> Generator[ServingRuntime]:
    """
    Provides a ServingRuntime resource for MLServer using the pre-installed template.

    Args:
        request (pytest.FixtureRequest): Pytest fixture request containing parameters.
        admin_client (DynamicClient): Kubernetes dynamic client.
        model_namespace (Namespace): Kubernetes namespace for model deployment.
        mlserver_runtime_image (str): The container image for the MLServer runtime.

    Yields:
        ServingRuntime: An instance of the MLServer ServingRuntime configured as per parameters.
    """
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=ModelInferenceRuntime.MLSERVER_RUNTIME,
        namespace=model_namespace.name,
        template_name=RuntimeTemplates.MLSERVER,
        deployment_type=request.param["deployment_type"],
        runtime_image=mlserver_runtime_image,
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def mlserver_inference_service(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    mlserver_model_service_account: ServiceAccount,
) -> Generator[InferenceService]:
    """
    Creates and yields a configured InferenceService instance for MLServer testing.

    Args:
        request (pytest.FixtureRequest): Pytest fixture request containing test parameters.
        admin_client (DynamicClient): Kubernetes dynamic client.
        model_namespace (Namespace): Kubernetes namespace for model deployment.
        mlserver_serving_runtime (ServingRuntime): The MLServer ServingRuntime instance.
        s3_models_storage_uri (str): URI for the S3 storage location of models.
        mlserver_model_service_account (ServiceAccount): Service account for the model.

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
        "model_format": mlserver_serving_runtime.instance.spec.supportedModelFormats[0].name,
        "model_service_account": mlserver_model_service_account.name,
        "deployment_mode": params.get("deployment_type", KServeDeploymentType.RAW_DEPLOYMENT),
        "external_route": params.get("enable_external_route", False),
    }

    gpu_count = params.get("gpu_count", 0)
    timeout = params.get("timeout")
    min_replicas = params.get("min-replicas")

    resources = copy.deepcopy(cast(dict[str, dict[str, str]], PREDICT_RESOURCES["resources"]))
    if gpu_count > 0:
        identifier = Labels.Nvidia.NVIDIA_COM_GPU
        resources["requests"][identifier] = gpu_count
        resources["limits"][identifier] = gpu_count
        service_config["volumes"] = copy.deepcopy(PREDICT_RESOURCES["volumes"])
        service_config["volumes_mounts"] = copy.deepcopy(PREDICT_RESOURCES["volume_mounts"])
    service_config["resources"] = resources

    if timeout:
        service_config["timeout"] = timeout

    if min_replicas:
        service_config["min_replicas"] = min_replicas

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
