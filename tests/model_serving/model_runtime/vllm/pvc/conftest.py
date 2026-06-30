from collections.abc import Generator
from copy import deepcopy
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime
from pytest import FixtureRequest

from tests.model_serving.model_runtime.vllm.constant import ACCELERATOR_IDENTIFIER, PREDICT_RESOURCES
from tests.model_serving.model_runtime.vllm.utils import (
    add_image_pull_secrets_if_configured,
    dedupe_vllm_cli_args,
    get_gpu_node_zone_selector,
    validate_supported_quantization_schema,
)
from utilities.constants import KServeDeploymentType, Labels
from utilities.general import download_model_data
from utilities.inference_utils import create_isvc


@pytest.fixture(scope="class")
def vllm_model_pvc(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    """PVC for storing vLLM model data downloaded from S3."""
    pvc_kwargs: dict[str, Any] = {
        "name": "vllm-model-pvc",
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
def pvc_downloaded_model_data(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    vllm_model_pvc: PersistentVolumeClaim,
    supported_accelerator_type: str,
    aws_secret_access_key: str,
    aws_access_key_id: str,
    models_s3_bucket_name: str,
    models_s3_bucket_endpoint: str,
    models_s3_bucket_region: str,
) -> str:
    """Download vLLM model data from the models S3 bucket into the PVC."""
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
        model_pvc_name=vllm_model_pvc.name,
        bucket_name=models_s3_bucket_name,
        aws_endpoint_url=models_s3_bucket_endpoint,
        aws_default_region=models_s3_bucket_region,
        model_path=request.param["model-dir"],
        use_sub_path=True,
        restricted_scc_init=True,
        node_selector=node_selector,
    )


@pytest.fixture(scope="class")
def vllm_pvc_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    serving_runtime: ServingRuntime,
    supported_accelerator_type: str,
    vllm_model_pvc: PersistentVolumeClaim,
    pvc_downloaded_model_data: str,
    kserve_registry_pull_secret: Secret | None,
) -> Generator[InferenceService, Any, Any]:
    """vLLM InferenceService backed by PVC storage."""
    isvc_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": request.param["name"],
        "namespace": model_namespace.name,
        "runtime": serving_runtime.name,
        "storage_uri": f"pvc://{vllm_model_pvc.name}/{pvc_downloaded_model_data}",
        "model_format": serving_runtime.instance.spec.supportedModelFormats[0].name,
        "deployment_mode": request.param.get("deployment_mode", KServeDeploymentType.STANDARD),
        "external_route": True,
    }

    accelerator_type = supported_accelerator_type.lower()
    raw_gpu_count = request.param.get("gpu_count")
    if raw_gpu_count is None:
        raise ValueError("gpu_count is required in request.param")
    try:
        gpu_count = int(raw_gpu_count)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"gpu_count must be an integer >= 0, got {raw_gpu_count!r}") from exc
    if gpu_count < 0:
        raise ValueError(f"gpu_count must be >= 0, got {gpu_count}")

    timeout = request.param.get("timeout")
    identifier = ACCELERATOR_IDENTIFIER.get(accelerator_type, Labels.Nvidia.NVIDIA_COM_GPU)
    resources = deepcopy(x=PREDICT_RESOURCES["resources"])
    resources["requests"][identifier] = gpu_count
    resources["limits"][identifier] = gpu_count
    isvc_kwargs["resources"] = resources

    if timeout:
        isvc_kwargs["timeout"] = timeout

    if gpu_count > 1:
        isvc_kwargs["volumes"] = PREDICT_RESOURCES["volumes"]
        isvc_kwargs["volumes_mounts"] = PREDICT_RESOURCES["volume_mounts"]

    if arguments := request.param.get("runtime_argument"):
        arguments = [arg for arg in arguments if not arg.startswith(("--tensor-parallel-size", "--quantization"))]
        arguments.append(f"--tensor-parallel-size={gpu_count}")
        if quantization := request.param.get("quantization"):
            validate_supported_quantization_schema(q_type=quantization)
            arguments.append(f"--quantization={quantization}")
        isvc_kwargs["argument"] = dedupe_vllm_cli_args(arguments=arguments)

    if min_replicas := request.param.get("min-replicas"):
        isvc_kwargs["min_replicas"] = min_replicas

    add_image_pull_secrets_if_configured(
        isvc_kwargs=isvc_kwargs,
        kserve_registry_pull_secret=kserve_registry_pull_secret,
    )

    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc
