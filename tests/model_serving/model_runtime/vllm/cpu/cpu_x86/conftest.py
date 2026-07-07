from collections.abc import Generator
from copy import deepcopy
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime
from pytest import FixtureRequest

from tests.model_serving.model_runtime.vllm.constant import TEMPLATE_MAP
from tests.model_serving.model_runtime.vllm.cpu.cpu_x86.constant import (
    CPU_X86_PREDICT_RESOURCES,
    CPU_X86_VOLUME_MOUNTS,
    CPU_X86_VOLUMES,
)
from tests.model_serving.model_runtime.vllm.utils import (
    add_image_pull_secrets_if_configured,
    dedupe_vllm_cli_args,
    skip_if_not_deployment_mode,
    validate_supported_quantization_schema,
)
from utilities.constants import KServeDeploymentType, RuntimeTemplates, Timeout
from utilities.inference_utils import create_isvc
from utilities.serving_runtime import ServingRuntimeFromTemplate

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="class")
def cpu_x86_serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    supported_accelerator_type: str,
    vllm_runtime_image: str,
) -> Generator[ServingRuntime]:
    """ServingRuntime backed by the vLLM CPU x86 runtime template."""
    accelerator_type = supported_accelerator_type.lower()
    template_name = TEMPLATE_MAP.get(accelerator_type, RuntimeTemplates.VLLM_CPU_x86)
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="vllm-runtime",
        namespace=model_namespace.name,
        template_name=template_name,
        deployment_type=request.param["deployment_mode"],
        runtime_image=vllm_runtime_image,
        support_tgis_open_ai_endpoints=True,
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def cpu_x86_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    cpu_x86_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    vllm_model_service_account: Any,
    kserve_registry_pull_secret: Secret | None,
) -> Generator[InferenceService, Any, Any]:
    """vLLM InferenceService for CPU x86 deployments backed by S3 model storage."""
    isvc_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": request.param["name"],
        "namespace": model_namespace.name,
        "runtime": cpu_x86_serving_runtime.name,
        "storage_uri": s3_models_storage_uri,
        "model_format": cpu_x86_serving_runtime.instance.spec.supportedModelFormats[0].name,
        "model_service_account": vllm_model_service_account.name,
        "deployment_mode": request.param.get("deployment_mode", KServeDeploymentType.STANDARD),
        "external_route": True,
        "resources": deepcopy(x=CPU_X86_PREDICT_RESOURCES),
        "volumes": CPU_X86_VOLUMES,
        "volumes_mounts": CPU_X86_VOLUME_MOUNTS,
        "timeout": request.param.get("timeout", Timeout.TIMEOUT_20MIN),
    }

    if arguments := request.param.get("runtime_argument"):
        arguments = [arg for arg in arguments if not arg.startswith("--quantization")]
        if quantization := request.param.get("quantization"):
            validate_supported_quantization_schema(q_type=quantization)
            arguments.append(f"--quantization={quantization}")
        isvc_kwargs["argument"] = dedupe_vllm_cli_args(arguments=arguments)

    if min_replicas := request.param.get("min-replicas"):
        isvc_kwargs["min_replicas"] = min_replicas

    if model_env_variables := request.param.get("model_env_variables"):
        isvc_kwargs["model_env_variables"] = model_env_variables

    add_image_pull_secrets_if_configured(
        isvc_kwargs=isvc_kwargs,
        kserve_registry_pull_secret=kserve_registry_pull_secret,
    )

    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc


@pytest.fixture
def skip_if_not_cpu_x86_raw_deployment(cpu_x86_inference_service: InferenceService) -> None:
    skip_if_not_deployment_mode(
        isvc=cpu_x86_inference_service,
        deployment_types=KServeDeploymentType.RAW_DEPLOYMENT_MODES,
    )
