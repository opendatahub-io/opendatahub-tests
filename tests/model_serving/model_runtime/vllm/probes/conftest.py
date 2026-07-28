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

from tests.model_serving.model_runtime.vllm.constant import TEMPLATE_MAP
from tests.model_serving.model_runtime.vllm.cpu.cpu_x86.constant import (
    CPU_X86_PREDICT_RESOURCES,
    CPU_X86_VOLUME_MOUNTS,
    CPU_X86_VOLUMES,
)
from tests.model_serving.model_runtime.vllm.probes.utils import VLLM_LIVENESS_PROBE, VLLM_READINESS_PROBE
from tests.model_serving.model_runtime.vllm.utils import (
    add_image_pull_secrets_if_configured,
    dedupe_vllm_cli_args,
    skip_if_not_deployment_mode,
    validate_supported_quantization_schema,
)
from utilities.constants import Containers, KServeDeploymentType, RuntimeTemplates
from utilities.inference_utils import create_isvc
from utilities.infra import get_pods_by_isvc_label
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="class")
def probes_serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    supported_accelerator_type: str,
    vllm_runtime_image: str,
) -> Generator[ServingRuntime, Any, Any]:
    """vLLM CPU x86 ServingRuntime with readiness and liveness httpGet probes on kserve-container."""
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
        containers={
            Containers.KSERVE_CONTAINER_NAME: {
                "readinessProbe": VLLM_READINESS_PROBE,
                "livenessProbe": VLLM_LIVENESS_PROBE,
            }
        },
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def vllm_probes_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    probes_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    vllm_model_service_account: ServiceAccount,
    kserve_registry_pull_secret: Secret | None,
) -> Generator[InferenceService, Any, Any]:
    """vLLM CPU x86 InferenceService with probe-enabled runtime backed by S3 model storage."""
    isvc_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": request.param["name"],
        "namespace": model_namespace.name,
        "runtime": probes_serving_runtime.name,
        "storage_uri": s3_models_storage_uri,
        "model_format": probes_serving_runtime.instance.spec.supportedModelFormats[0].name,
        "model_service_account": vllm_model_service_account.name,
        "deployment_mode": request.param.get("deployment_mode", KServeDeploymentType.STANDARD),
        "external_route": True,
        "resources": deepcopy(x=CPU_X86_PREDICT_RESOURCES),
        "volumes": CPU_X86_VOLUMES,
        "volumes_mounts": CPU_X86_VOLUME_MOUNTS,
        "timeout": request.param.get("timeout", 1200),
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
def vllm_probes_pod_resource(admin_client: DynamicClient, vllm_probes_inference_service: InferenceService) -> Pod:
    return get_pods_by_isvc_label(client=admin_client, isvc=vllm_probes_inference_service)[0]


@pytest.fixture
def skip_if_not_probes_raw_deployment(vllm_probes_inference_service: InferenceService) -> None:
    skip_if_not_deployment_mode(
        isvc=vllm_probes_inference_service,
        deployment_types=KServeDeploymentType.RAW_DEPLOYMENT_MODES,
    )
