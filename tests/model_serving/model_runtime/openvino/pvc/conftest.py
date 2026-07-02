from collections.abc import Generator
from copy import deepcopy
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.serving_runtime import ServingRuntime
from pytest import FixtureRequest

from tests.model_serving.model_runtime.openvino.constant import PREDICT_RESOURCES
from utilities.constants import KServeDeploymentType, Labels
from utilities.general import download_model_data
from utilities.inference_utils import create_isvc
from utilities.infra import get_pods_by_isvc_label


@pytest.fixture(scope="class")
def openvino_model_pvc(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    """PVC for storing OpenVINO model data downloaded from S3."""
    pvc_kwargs: dict[str, Any] = {
        "name": "openvino-model-pvc",
        "namespace": model_namespace.name,
        "client": admin_client,
        "size": request.param["pvc-size"],
        "accessmodes": request.param.get("access-modes", "ReadWriteOnce"),
    }
    if storage_class_name := request.param.get("storage-class-name"):
        pvc_kwargs["storage_class"] = storage_class_name

    with PersistentVolumeClaim(**pvc_kwargs) as pvc:
        yield pvc


@pytest.fixture(scope="class")
def openvino_pvc_downloaded_model_data(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    openvino_model_pvc: PersistentVolumeClaim,
    aws_secret_access_key: str,
    aws_access_key_id: str,
    models_s3_bucket_name: str,
    models_s3_bucket_endpoint: str,
    models_s3_bucket_region: str,
) -> str:
    """Download OpenVINO model data from the models S3 bucket into the PVC."""
    return download_model_data(
        client=admin_client,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        model_namespace=model_namespace.name,
        model_pvc_name=openvino_model_pvc.name,
        bucket_name=models_s3_bucket_name,
        aws_endpoint_url=models_s3_bucket_endpoint,
        aws_default_region=models_s3_bucket_region,
        model_path=request.param["model-dir"],
        use_sub_path=True,
        restricted_scc_init=True,
    )


@pytest.fixture(scope="class")
def openvino_pvc_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    openvino_pvc_serving_runtime: ServingRuntime,
    gpu_count_on_cluster: int,
    openvino_model_pvc: PersistentVolumeClaim,
    openvino_pvc_downloaded_model_data: str,
) -> Generator[InferenceService, Any, Any]:
    """OpenVINO InferenceService backed by PVC storage."""
    isvc_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": request.param["name"],
        "namespace": model_namespace.name,
        "runtime": openvino_pvc_serving_runtime.name,
        "storage_uri": f"pvc://{openvino_model_pvc.name}/{openvino_pvc_downloaded_model_data}",
        "model_format": openvino_pvc_serving_runtime.instance.spec.supportedModelFormats[0].name,
        "deployment_mode": request.param.get("deployment_mode", KServeDeploymentType.STANDARD),
        "external_route": request.param.get("enable_external_route", False),
    }

    gpu_count = request.param.get("gpu_count", 0)
    timeout = request.param.get("timeout")
    min_replicas = request.param.get("min-replicas")

    resources = deepcopy(x=PREDICT_RESOURCES["resources"])
    if gpu_count > 0:
        if gpu_count_on_cluster < gpu_count:
            raise ResourceNotFoundError(
                f"Not enough GPU available for test execution, required:{gpu_count}, available:{gpu_count_on_cluster}"
            )
        identifier = Labels.Nvidia.NVIDIA_COM_GPU
        resources["requests"][identifier] = gpu_count
        resources["limits"][identifier] = gpu_count

        if gpu_count > 1:
            isvc_kwargs["volumes"] = PREDICT_RESOURCES["volumes"]
            isvc_kwargs["volumes_mounts"] = PREDICT_RESOURCES["volume_mounts"]

    isvc_kwargs["resources"] = resources

    if timeout:
        isvc_kwargs["timeout"] = timeout

    if min_replicas:
        isvc_kwargs["min_replicas"] = min_replicas

    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def openvino_pvc_pod_resource(
    admin_client: DynamicClient,
    openvino_pvc_inference_service: InferenceService,
) -> Pod:
    """Get the OpenVINO pod for the PVC-backed InferenceService."""
    pods = get_pods_by_isvc_label(client=admin_client, isvc=openvino_pvc_inference_service)
    if not pods:
        raise ResourceNotFoundError(f"No pods found for InferenceService {openvino_pvc_inference_service.name}")
    return pods[0]
