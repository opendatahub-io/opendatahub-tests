from typing import Any, Generator
import pytest
from tests.model_serving.model_runtime.model_validation.constant import (
    ACCELERATOR_IDENTIFIER,
    TEMPLATE_MAP,
    PREDICT_RESOURCES,
)
from utilities.constants import KServeDeploymentType, Labels, RuntimeTemplates
from ocp_resources.namespace import Namespace
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.inference_service import InferenceService
from pytest import FixtureRequest
from pytest_testconfig import config as py_config
from kubernetes.dynamic import DynamicClient
from ocp_resources.cluster_service_version import ClusterServiceVersion
from simple_logger.logger import get_logger
from utilities.serving_runtime import ServingRuntimeFromTemplate
from utilities.inference_utils import create_isvc
from tests.model_serving.model_runtime.vllm.utils import validate_supported_quantization_schema
from tests.model_serving.model_runtime.model_validation.constant import (
    ORIGINAL_PULL_SECRET,
    INFERENCE_SERVICE_PORT,
    CONTAINER_PORT,
)

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="package")
def fail_if_missing_dependent_operators(admin_client: DynamicClient) -> None:
    if dependent_operators := py_config.get("dependent_operators"):
        missing_operators: list[str] = []

        for operator_name in dependent_operators.split(","):
            csvs = list(
                ClusterServiceVersion.get(
                    dyn_client=admin_client,
                    namespace=py_config["applications_namespace"],
                )
            )

            LOGGER.info(f"Verifying if {operator_name} is installed")
            for csv in csvs:
                if csv.name.startswith(operator_name):
                    if csv.status == csv.Status.SUCCEEDED:
                        break

                    else:
                        missing_operators.append(
                            f"Operator {operator_name} is installed but CSV is not in {csv.Status.SUCCEEDED} state"
                        )

            else:
                missing_operators.append(f"{operator_name} is not installed")

        if missing_operators:
            pytest.fail(str(missing_operators))

    else:
        LOGGER.info("No dependent operators to verify")


@pytest.fixture(scope="session")
def skip_if_no_supported_accelerator_type(supported_accelerator_type: str) -> None:
    if not supported_accelerator_type:
        pytest.skip("Accelartor type is not provided,vLLM test can not be run on CPU")


@pytest.fixture(scope="class")
def vllm_model_car_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    serving_runtime: ServingRuntime,
    supported_accelerator_type: str,
    modelcar_image_uri: str,
) -> Generator[InferenceService, Any, Any]:
    isvc_kwargs = {
        "client": admin_client,
        "name": request.param["name"],
        "namespace": model_namespace.name,
        "runtime": serving_runtime.name,
        "storage_uri": modelcar_image_uri,
        "model_format": serving_runtime.instance.spec.supportedModelFormats[0].name,
        "deployment_mode": request.param.get("deployment_mode", KServeDeploymentType.SERVERLESS),
        "image_pull_secrets": [ORIGINAL_PULL_SECRET],
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

    if min_replicas := request.param.get("min-replicas"):
        isvc_kwargs["min_replicas"] = min_replicas

    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    supported_accelerator_type: str,
    vllm_runtime_image: str,
) -> Generator[ServingRuntime, None, None]:
    accelerator_type = supported_accelerator_type.lower()
    template_name = TEMPLATE_MAP.get(accelerator_type, RuntimeTemplates.VLLM_CUDA)
    print(f"using template: {template_name}")
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="vllm-runtime",
        namespace=model_namespace.name,
        template_name=template_name,
        deployment_type=request.param["deployment_type"],
        runtime_image=vllm_runtime_image,
        support_tgis_open_ai_endpoints=True,
        containers={
            "kserve-container": {
                "args": [
                    f"--port={str(INFERENCE_SERVICE_PORT)}",
                    "--model=/mnt/models",
                    "--served-model-name={{.Name}}",
                ],
                "ports": [
                    {
                        "containerPort": CONTAINER_PORT,
                        "protocol": "TCP",
                    }
                ],
                "volumeMounts": [{"mountPath": "/dev/shm", "name": "shm"}],
            }
        },
        volumes=[{"emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"}, "name": "shm"}],
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="session")
def modelcar_image_uri(request: FixtureRequest, model_image_name: str) -> str:
    if not model_image_name:
        return None
    return f"oci://registry.redhat.io/rhelai1/{model_image_name}"
