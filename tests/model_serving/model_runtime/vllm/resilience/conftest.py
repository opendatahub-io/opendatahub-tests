from collections.abc import Generator
from copy import deepcopy
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from pytest import FixtureRequest

from tests.model_serving.model_runtime.vllm.constant import ACCELERATOR_IDENTIFIER, PREDICT_RESOURCES, TEMPLATE_MAP
from tests.model_serving.model_runtime.vllm.utils import (
    add_image_pull_secrets_if_configured,
    dedupe_vllm_cli_args,
    validate_supported_quantization_schema,
)
from utilities.constants import KServeDeploymentType, Labels, RuntimeTemplates
from utilities.inference_utils import create_isvc
from utilities.infra import get_pods_by_isvc_label
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="class")
def resilience_serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    supported_accelerator_type: str,
    vllm_runtime_image: str,
) -> Generator[ServingRuntime]:
    accelerator_type = supported_accelerator_type.lower()
    template_name = TEMPLATE_MAP.get(accelerator_type, RuntimeTemplates.VLLM_CUDA)
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="vllm-res-runtime",
        namespace=model_namespace.name,
        template_name=template_name,
        deployment_type=request.param.get("deployment_mode", KServeDeploymentType.RAW_DEPLOYMENT),
        runtime_image=vllm_runtime_image,
        support_tgis_open_ai_endpoints=True,
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def resilience_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    resilience_serving_runtime: ServingRuntime,
    supported_accelerator_type: str,
    s3_models_storage_uri: str,
    vllm_model_service_account: ServiceAccount,
    kserve_registry_pull_secret: Secret | None,
) -> Generator[InferenceService, Any, Any]:
    isvc_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": request.param["name"],
        "namespace": model_namespace.name,
        "runtime": resilience_serving_runtime.name,
        "storage_uri": s3_models_storage_uri,
        "model_format": resilience_serving_runtime.instance.spec.supportedModelFormats[0].name,
        "model_service_account": vllm_model_service_account.name,
        "deployment_mode": request.param.get("deployment_mode", KServeDeploymentType.RAW_DEPLOYMENT),
        "external_route": True,
    }

    accelerator_type = supported_accelerator_type.lower()
    gpu_count = request.param.get("gpu_count", 1)
    identifier = ACCELERATOR_IDENTIFIER.get(accelerator_type, Labels.Nvidia.NVIDIA_COM_GPU)
    resources = deepcopy(PREDICT_RESOURCES["resources"])
    resources["requests"][identifier] = gpu_count
    resources["limits"][identifier] = gpu_count
    isvc_kwargs["resources"] = resources

    if timeout := request.param.get("timeout"):
        isvc_kwargs["timeout"] = timeout

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


@pytest.fixture
def resilience_pod_resource(
    admin_client: DynamicClient,
    resilience_inference_service: InferenceService,
) -> Pod:
    return get_pods_by_isvc_label(client=admin_client, isvc=resilience_inference_service)[0]
