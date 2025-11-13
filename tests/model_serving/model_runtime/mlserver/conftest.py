"""
Pytest fixtures for MLServer model serving runtime tests.

This module provides fixtures for:
- Setting up MLServer serving runtimes and templates
- Creating inference services and related Kubernetes resources
- Managing S3 secrets and service accounts
- Providing test utilities like snapshots and pod resources
"""

from typing import cast, Any, Generator
import copy

import pytest
from syrupy.extensions.json import JSONSnapshotExtension
from pytest_testconfig import config as py_config

from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.template import Template
from ocp_resources.service_account import ServiceAccount

from tests.model_serving.model_runtime.mlserver.constant import (
    PREDICT_RESOURCES,
    RUNTIME_NAME_MAP,
    TEMPLATE_NAME_MAP,
)

from tests.model_serving.model_runtime.mlserver.utils import mlserver_runtime_template_dict

from utilities.constants import (
    KServeDeploymentType,
    Labels,
    Protocols,
)
from utilities.inference_utils import create_isvc
from utilities.infra import get_pods_by_isvc_label
from utilities.serving_runtime import ServingRuntimeFromTemplate

from simple_logger.logger import get_logger


LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="class")
def mlserver_rest_serving_runtime_template(admin_client: DynamicClient) -> Generator[Template, None, None]:
    """
    Provides a REST serving runtime Template for MLServer within the test class scope.

    Args:
        admin_client (DynamicClient): Kubernetes dynamic client.

    Yields:
        Template: The loaded REST serving runtime Template.
    """
    with Template(
        client=admin_client,
        kind_dict=mlserver_runtime_template_dict(Protocols.REST),
        namespace=py_config["applications_namespace"],
    ) as tp:
        yield tp


@pytest.fixture(scope="class")
def mlserver_serving_runtime(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_runtime_image: str,
    protocol: str,
) -> Generator[ServingRuntime, None, None]:
    """
    Provides a ServingRuntime resource for MLServer with the specified protocol and deployment type.

    Args:
        request (pytest.FixtureRequest): Pytest fixture request containing parameters.
        admin_client (DynamicClient): Kubernetes dynamic client.
        model_namespace (Namespace): Kubernetes namespace for model deployment.
        mlserver_runtime_image (str): The container image for the MLServer runtime.
        protocol (str): The protocol to use (e.g., REST).

    Yields:
        ServingRuntime: An instance of the MLServer ServingRuntime configured as per parameters.
    """
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=RUNTIME_NAME_MAP.get(protocol, f"mlserver-{protocol}-runtime"),
        namespace=model_namespace.name,
        template_name=TEMPLATE_NAME_MAP.get(protocol, f"mlserver-{protocol}-template"),
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
) -> Generator[InferenceService, Any, Any]:
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
def mlserver_model_service_account(admin_client: DynamicClient, kserve_s3_secret: Secret) -> ServiceAccount:
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
