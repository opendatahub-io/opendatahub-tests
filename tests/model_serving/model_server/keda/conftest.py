from typing import Any, Generator

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from simple_logger.logger import get_logger
from tests.model_serving.model_runtime.vllm.utils import (
    validate_supported_quantization_schema,
    kserve_s3_endpoint_secret,
)
from tests.model_serving.model_runtime.vllm.constant import ACCELERATOR_IDENTIFIER, PREDICT_RESOURCES, TEMPLATE_MAP

from utilities.constants import (
    KServeDeploymentType,
    RuntimeTemplates,
    Labels,
)
from utilities.constants import (
    ModelAndFormat,
)
from utilities.inference_utils import create_isvc
from utilities.serving_runtime import ServingRuntimeFromTemplate
from utilities.constants import THANOS_QUERIER_ADDRESS
from syrupy.extensions.json import JSONSnapshotExtension

LOGGER = get_logger(name=__name__)


def create_keda_auto_scaling_config(
    query: str,
    model_name: str,
    namespace: str,
    target_value: str,
) -> dict[str, Any]:
    """Create KEDA auto-scaling configuration for inference services.

    Args:
        query: The Prometheus query to use for scaling
        model_name: Name of the model
        namespace: Kubernetes namespace
        target_value: Target value for the metric

    Returns:
        dict: Auto-scaling configuration
    """
    return {
        "metrics": [
            {
                "type": "External",
                "external": {
                    "metric": {
                        "backend": "prometheus",
                        "serverAddress": THANOS_QUERIER_ADDRESS,
                        "query": query,
                        "namespace": namespace,
                        "authModes": "bearer",
                    },
                    "authenticationRef": {
                        # https://github.com/opendatahub-io/odh-model-controller/blob/6c83a07e764be8a6bb77b1b36589ed029605b186/internal/controller/serving/reconcilers/kserve_keda_reconciler.go#L47
                        "name": "inference-prometheus-auth",
                    },
                    "target": {"type": "Value", "value": target_value},
                },
            }
        ]
    }


@pytest.fixture(scope="class")
def vllm_serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    supported_accelerator_type: str,
    vllm_runtime_image: str,
) -> Generator[ServingRuntime, None, None]:
    accelerator_type = supported_accelerator_type.lower()
    template_name = TEMPLATE_MAP.get(accelerator_type, RuntimeTemplates.VLLM_CUDA)
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="vllm-runtime",
        namespace=model_namespace.name,
        template_name=template_name,
        deployment_type=request.param["deployment_type"],
        runtime_image=vllm_runtime_image,
        support_tgis_open_ai_endpoints=True,
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def keda_vllm_model_service_account(
    admin_client: DynamicClient,
    keda_kserve_endpoint_s3_secret: Secret,
) -> Generator[ServiceAccount, None, None]:
    with ServiceAccount(
        client=admin_client,
        namespace=keda_kserve_endpoint_s3_secret.namespace,
        name="models-bucket-sa",
        secrets=[{"name": keda_kserve_endpoint_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def keda_kserve_endpoint_s3_secret(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
) -> Secret:
    with kserve_s3_endpoint_secret(
        admin_client=admin_client,
        name="models-bucket-secret",
        namespace=model_namespace.name,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_region=models_s3_bucket_region,
        aws_s3_endpoint=models_s3_bucket_endpoint,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def keda_vllm_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    vllm_serving_runtime: ServingRuntime,
    supported_accelerator_type: str,
    s3_models_storage_uri: str,
    keda_vllm_model_service_account: ServiceAccount,
) -> Generator[InferenceService, Any, Any]:
    isvc_kwargs = {
        "client": admin_client,
        "name": request.param["name"],
        "namespace": model_namespace.name,
        "runtime": vllm_serving_runtime.name,
        "storage_uri": s3_models_storage_uri,
        "model_format": vllm_serving_runtime.instance.spec.supportedModelFormats[0].name,
        "model_service_account": keda_vllm_model_service_account.name,
        "deployment_mode": request.param.get("deployment_mode", KServeDeploymentType.RAW_DEPLOYMENT),
        "autoscaler_mode": "keda",
        "external_route": True,
    }
    accelerator_type = supported_accelerator_type.lower()
    gpu_count = request.param.get("gpu_count")
    timeout = request.param.get("timeout")
    identifier = ACCELERATOR_IDENTIFIER.get(accelerator_type, Labels.Nvidia.NVIDIA_COM_GPU)
    resources: Any = PREDICT_RESOURCES["resources"]
    resources["requests"][identifier] = gpu_count
    resources["limits"][identifier] = gpu_count
    isvc_kwargs["resources"] = resources
    if timeout:
        isvc_kwargs["timeout"] = timeout
    if gpu_count > 1:
        isvc_kwargs["volumes"] = PREDICT_RESOURCES["volumes"]
        isvc_kwargs["volumes_mounts"] = PREDICT_RESOURCES["volume_mounts"]
    if arguments := request.param.get("runtime_argument"):
        arguments = [
            arg
            for arg in arguments
            if not (arg.startswith("--tensor-parallel-size") or arg.startswith("--quantization"))
        ]
        arguments.append(f"--tensor-parallel-size={gpu_count}")
        if quantization := request.param.get("quantization"):
            validate_supported_quantization_schema(q_type=quantization)
            arguments.append(f"--quantization={quantization}")
        isvc_kwargs["argument"] = arguments

    isvc_kwargs["min_replicas"] = request.param.get("initial_pod_count")
    isvc_kwargs["max_replicas"] = request.param.get("final_pod_count")

    isvc_kwargs["auto_scaling"] = create_keda_auto_scaling_config(
        query=request.param.get("metrics_query"),
        model_name=request.param["name"],
        namespace=model_namespace.name,
        target_value=str(request.param.get("metrics_threshold")),
    )

    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def ovms_keda_inference_service(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    ovms_kserve_serving_runtime: ServingRuntime,
    ci_endpoint_s3_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    model_name = f"{request.param['name']}-raw"
    with create_isvc(
        client=unprivileged_client,
        name=model_name,
        namespace=unprivileged_model_namespace.name,
        external_route=True,
        runtime=ovms_kserve_serving_runtime.name,
        storage_path=request.param["model-dir"],
        storage_key=ci_endpoint_s3_secret.name,
        model_format=ModelAndFormat.OPENVINO_IR,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        model_version=request.param["model-version"],
        min_replicas=request.param["initial_pod_count"],
        max_replicas=request.param["final_pod_count"],
        autoscaler_mode="keda",
        auto_scaling=create_keda_auto_scaling_config(
            query=request.param["metrics_query"],
            model_name=model_name,
            namespace=unprivileged_model_namespace.name,
            target_value=str(request.param["metrics_threshold"]),
        ),
    ) as isvc:
        yield isvc


@pytest.fixture(scope="session")
def skip_if_no_supported_gpu_type(supported_accelerator_type: str) -> None:
    if not supported_accelerator_type:
        pytest.skip("Accelartor type is not provide,vLLM test can not be run on CPU")


@pytest.fixture
def response_snapshot(snapshot: Any) -> Any:
    return snapshot.use_extension(extension_class=JSONSnapshotExtension)
