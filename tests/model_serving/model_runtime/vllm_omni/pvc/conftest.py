"""Fixtures for vLLM-Omni PVC tests."""

from collections.abc import Generator
from copy import deepcopy
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.serving_runtime import ServingRuntime
from pytest import FixtureRequest

from tests.model_serving.model_runtime.vllm.constant import ACCELERATOR_IDENTIFIER
from tests.model_serving.model_runtime.vllm.utils import dedupe_vllm_cli_args, get_gpu_node_zone_selector
from tests.model_serving.model_runtime.vllm_omni.constant import (
    OMNI_MULTI_GPU_RESOURCES,
    OMNI_SERVING_ARGS,
    OMNI_SINGLE_GPU_RESOURCES,
    OMNI_VOLUME_MOUNTS,
    OMNI_VOLUMES,
)
from utilities.constants import KServeDeploymentType, Labels
from utilities.general import download_model_data
from utilities.inference_utils import create_isvc


@pytest.fixture(scope="class")
def vllm_omni_model_pvc(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    """PVC for storing vLLM-Omni model data downloaded from S3."""
    pvc_kwargs: dict[str, Any] = {
        "name": "vllm-omni-model-pvc",
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
def vllm_omni_pvc_downloaded_model_data(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    vllm_omni_model_pvc: PersistentVolumeClaim,
    supported_accelerator_type: str,
    aws_secret_access_key: str,
    aws_access_key_id: str,
    models_s3_bucket_name: str,
    models_s3_bucket_endpoint: str,
    models_s3_bucket_region: str,
) -> str:
    """Download vLLM-Omni model data from the models S3 bucket into the PVC."""
    gpu_resource = ACCELERATOR_IDENTIFIER.get(
        supported_accelerator_type.lower(),
        Labels.Nvidia.NVIDIA_COM_GPU,
    )
    node_selector = get_gpu_node_zone_selector(client=admin_client, gpu_resource=gpu_resource)
    return download_model_data(
        client=admin_client,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        model_namespace=model_namespace.name,
        model_pvc_name=vllm_omni_model_pvc.name,
        bucket_name=models_s3_bucket_name,
        aws_endpoint_url=models_s3_bucket_endpoint,
        aws_default_region=models_s3_bucket_region,
        model_path=request.param["model-dir"],
        use_sub_path=True,
        restricted_scc_init=True,
        node_selector=node_selector,
    )


@pytest.fixture(scope="class")
def vllm_omni_pvc_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    vllm_omni_serving_runtime: ServingRuntime,
    supported_accelerator_type: str,
    vllm_omni_model_pvc: PersistentVolumeClaim,
    vllm_omni_pvc_downloaded_model_data: str,
) -> Generator[InferenceService, Any, Any]:
    """vLLM-Omni InferenceService backed by PVC storage."""
    accelerator_type = supported_accelerator_type.lower()
    raw_gpu_count = request.param.get("gpu_count")
    if raw_gpu_count is None:
        raise ValueError("gpu_count is required in vllm_omni_pvc_inference_service request.param")
    try:
        gpu_count = int(raw_gpu_count)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"gpu_count must be an integer >= 0, got {raw_gpu_count!r}") from exc
    if gpu_count < 0:
        raise ValueError(f"gpu_count must be >= 0, got {gpu_count}")
    identifier = ACCELERATOR_IDENTIFIER.get(accelerator_type, Labels.Nvidia.NVIDIA_COM_GPU)

    base_resources = OMNI_MULTI_GPU_RESOURCES if gpu_count > 1 else OMNI_SINGLE_GPU_RESOURCES
    resources = deepcopy(x=base_resources["resources"])
    resources["requests"][identifier] = gpu_count
    resources["limits"][identifier] = gpu_count

    serving_args = list(OMNI_SERVING_ARGS)

    isvc_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": request.param["name"],
        "namespace": model_namespace.name,
        "runtime": vllm_omni_serving_runtime.name,
        "storage_uri": f"pvc://{vllm_omni_model_pvc.name}/{vllm_omni_pvc_downloaded_model_data}",
        "model_format": vllm_omni_serving_runtime.instance.spec.supportedModelFormats[0].name,
        "deployment_mode": request.param.get("deployment_mode", KServeDeploymentType.STANDARD),
        "external_route": True,
        "resources": resources,
        "argument": dedupe_vllm_cli_args(serving_args),
    }

    if gpu_count > 1:
        isvc_kwargs["volumes"] = OMNI_VOLUMES
        isvc_kwargs["volumes_mounts"] = OMNI_VOLUME_MOUNTS

    if timeout := request.param.get("timeout"):
        isvc_kwargs["timeout"] = timeout

    if min_replicas := request.param.get("min-replicas"):
        isvc_kwargs["min_replicas"] = min_replicas

    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc
