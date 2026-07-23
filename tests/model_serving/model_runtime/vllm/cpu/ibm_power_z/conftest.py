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
from tests.model_serving.model_runtime.vllm.cpu.ibm_power_z.constant import IBM_POWER_Z_PREDICT_RESOURCES
from tests.model_serving.model_runtime.vllm.utils import (
    add_image_pull_secrets_if_configured,
    dedupe_vllm_cli_args,
    skip_if_not_deployment_mode,
    validate_supported_quantization_schema,
)
from utilities.constants import AcceleratorType, KServeDeploymentType, RuntimeTemplates, Timeout
from utilities.inference_utils import create_isvc
from utilities.serving_runtime import ServingRuntimeFromTemplate

LOGGER = structlog.get_logger(name=__name__)

SUPPORTED_IBM_POWER_Z_ACCELERATORS: set[str] = {
    AcceleratorType.CPU_POWER,
    AcceleratorType.CPU_Z,
}


@pytest.fixture(scope="session")
def skip_if_no_supported_ibm_power_z_accelerator_type(supported_accelerator_type: str | None) -> None:
    """Skip test unless the cluster provides a supported IBM Power or Z CPU accelerator."""
    if not supported_accelerator_type or supported_accelerator_type.lower() not in SUPPORTED_IBM_POWER_Z_ACCELERATORS:
        pytest.skip(
            f"Test requires a supported vLLM IBM Power or Z CPU accelerator. "
            f"Found: '{supported_accelerator_type or 'None'}'. "
            f"Expected one of: {SUPPORTED_IBM_POWER_Z_ACCELERATORS}."
        )


@pytest.fixture(scope="class")
def ibm_power_z_serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    skip_if_no_supported_ibm_power_z_accelerator_type: None,
    supported_accelerator_type: str,
    vllm_runtime_image: str,
) -> Generator[ServingRuntime]:
    """ServingRuntime backed by the vLLM CPU Power/Z runtime template.

    Both cpu_power and cpu_z accelerator types share a single cluster template
    (vllm-cpu-runtime-template) on ppc64le/s390x clusters.
    """
    accelerator_type = supported_accelerator_type.lower()
    template_name = TEMPLATE_MAP.get(accelerator_type, RuntimeTemplates.VLLM_CPU_POWER)
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
def ibm_power_z_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    ibm_power_z_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    vllm_model_service_account: Any,
    kserve_registry_pull_secret: Secret | None,
) -> Generator[InferenceService, Any, Any]:
    """vLLM InferenceService for CPU Power or Z deployments backed by S3 model storage."""
    isvc_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": request.param["name"],
        "namespace": model_namespace.name,
        "runtime": ibm_power_z_serving_runtime.name,
        "storage_uri": s3_models_storage_uri,
        "model_format": ibm_power_z_serving_runtime.instance.spec.supportedModelFormats[0].name,
        "model_service_account": vllm_model_service_account.name,
        "deployment_mode": request.param.get("deployment_mode", KServeDeploymentType.STANDARD),
        "external_route": True,
        "resources": deepcopy(x=request.param.get("resources", IBM_POWER_Z_PREDICT_RESOURCES)),
        "timeout": request.param.get("timeout", Timeout.TIMEOUT_30MIN),
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
def skip_if_not_ibm_power_z_raw_deployment(ibm_power_z_inference_service: InferenceService) -> None:
    skip_if_not_deployment_mode(
        isvc=ibm_power_z_inference_service,
        deployment_types=KServeDeploymentType.RAW_DEPLOYMENT_MODES,
    )
