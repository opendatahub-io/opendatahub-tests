import json
from typing import Any, Generator, List

import pytest
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime
from pytest import FixtureRequest
from utilities.infra import get_pods_by_isvc_label

from tests.model_serving.model_runtime.model_validation.constant import (
    ACCELERATOR_IDENTIFIER,
    TEMPLATE_MAP,
    PREDICT_RESOURCES,
    PULL_SECRET_ACCESS_TYPE,
)
from tests.model_serving.model_runtime.model_validation.constant import (
    BASE_RAW_DEPLOYMENT_CONFIG,
)
from tests.model_serving.model_runtime.model_validation.constant import PULL_SECRET_NAME
from tests.model_serving.model_runtime.model_validation.constant import (
    TIMEOUT_20MIN,
)
from tests.model_serving.model_runtime.model_validation.utils import safe_k8s_name
from tests.model_serving.model_runtime.vllm.utils import validate_supported_quantization_schema
from utilities.constants import KServeDeploymentType, Labels, RuntimeTemplates
from utilities.inference_utils import create_isvc
from utilities.serving_runtime import ServingRuntimeFromTemplate

from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="class")
def model_car_serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    supported_accelerator_type: str,
    vllm_runtime_image: str,
) -> Generator[ServingRuntime, None, None]:
    accelerator_type = supported_accelerator_type.lower()

    template_name = TEMPLATE_MAP.get(accelerator_type, RuntimeTemplates.VLLM_CUDA)
    LOGGER.info(f"using template: {template_name}")
    assert model_namespace.name is not None
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=f"vllm-{request.param['deployment_type'].lower()}-runtime",
        namespace=model_namespace.name,
        template_name=template_name,
        deployment_type=request.param["deployment_type"],
        runtime_image=vllm_runtime_image,
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def vllm_model_car_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    model_car_serving_runtime: ServingRuntime,
    supported_accelerator_type: str,
    deployment_config: dict[str, Any],
    kserve_registry_pull_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    isvc_kwargs = {
        "client": admin_client,
        "name": safe_k8s_name(request.param.get("model_name", "")),
        "namespace": model_namespace.name,
        "runtime": model_car_serving_runtime.name,
        "storage_uri": request.param.get("model_car_image_uri"),
        "model_format": model_car_serving_runtime.instance.spec.supportedModelFormats[0].name,
        "deployment_mode": deployment_config.get("deployment_type", KServeDeploymentType.RAW_DEPLOYMENT),
        "image_pull_secrets": [kserve_registry_pull_secret.name],
    }
    accelerator_type = supported_accelerator_type.lower()
    gpu_count = deployment_config.get("gpu_count", 0)
    timeout = deployment_config.get("timeout")
    identifier = ACCELERATOR_IDENTIFIER.get(accelerator_type, Labels.Nvidia.NVIDIA_COM_GPU)
    resources: Any = PREDICT_RESOURCES["resources"]
    resources["requests"][identifier] = gpu_count
    resources["limits"][identifier] = gpu_count
    isvc_kwargs["resources"] = resources

    if identifier == Labels.Spyre.SPYRE_COM_GPU:
        isvc_kwargs["scheduler_name"] = "spyre-scheduler"
        resources["requests"] = {
            "ibm.com/spyre_pf": gpu_count,
        }
        resources["limits"] = {
            "ibm.com/spyre_pf": gpu_count,
        }

    if timeout:
        isvc_kwargs["timeout"] = timeout

    if gpu_count > 1:
        isvc_kwargs["volumes"] = PREDICT_RESOURCES["volumes"]
        isvc_kwargs["volumes_mounts"] = PREDICT_RESOURCES["volume_mounts"]

    if arguments := deployment_config.get("runtime_argument"):
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
def kserve_registry_pull_secret(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    registry_pull_secret: list[str],
    registry_host: list[str],
) -> Generator[Secret, Any, Any]:
    docker_config_json = json.dumps(
        {"auths": {host: {"auth": auth} for host, auth in zip(registry_host, registry_pull_secret)}}
        )
    with Secret(
        client=admin_client,
        name=PULL_SECRET_NAME,
        namespace=model_namespace.name,
        string_data={
            ".dockerconfigjson": docker_config_json,
            "ACCESS_TYPE": PULL_SECRET_ACCESS_TYPE,
            "OCI_HOST": ",".join(registry_host),
        },
        type="kubernetes.io/dockerconfigjson",
        wait_for_resource=True,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def deployment_config(request: FixtureRequest) -> dict[str, Any]:
    """
    Fixture to provide the base deployment configuration for raw deployments.
    """
    deployment_type = request.param.get("deployment_type", KServeDeploymentType.RAW_DEPLOYMENT)
    serving_argument = request.param.get("runtime_argument", [])

    config = BASE_RAW_DEPLOYMENT_CONFIG.copy()
    config["runtime_argument"] = serving_argument
    config["deployment_type"] = deployment_type
    config["gpu_count"] = request.param.get("gpu_count", 1)
    config["model_output_type"] = request.param.get("model_output_type", "text")
    config["timeout"] = TIMEOUT_20MIN
    return config


def build_raw_params(
    name: str,
    image: str,
    args: list[str],
    gpu_count: int,
    execution_mode: str,
    model_output_type: str = "text",
) -> tuple[Any, str]:
    test_id = f"{name}-raw"
    deployment_type = KServeDeploymentType.RAW_DEPLOYMENT
    param = pytest.param(
        {"name": "raw-model-validation"},
        {"deployment_type": deployment_type},
        {
            "model_name": name,
            "model_car_image_uri": image,
        },
        {
            "deployment_type": deployment_type,
            "runtime_argument": args,
            "gpu_count": gpu_count,
            "model_output_type": model_output_type,
        },
        id=test_id,
        marks=build_pytest_markers(deployment_type=deployment_type, execution_mode=execution_mode),
    )
    return param, test_id


def build_pytest_markers(deployment_type: str, execution_mode: str) -> List[Any]:
    """
    Build a list of pytest markers based on deployment type, execution mode.

    Args:
        deployment_type (str): Deployment type (e.g., RAW_DEPLOYMENT)
        execution_mode (str): "parallel" or "sequential"

    Returns:
        List[Any]: List of pytest.mark objects to attach to the test
    """
    markers: List[pytest.MarkDecorator] = []

    if deployment_type == KServeDeploymentType.RAW_DEPLOYMENT:
        markers.append(pytest.mark.rawdeployment)

    # Execution mode markers
    if execution_mode == "parallel":
        markers.append(pytest.mark.parallel)
        markers.append(pytest.mark.skip_must_gather)

    return markers


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    yaml_config = None
    yaml_path = metafunc.config.getoption(name="model_car_yaml_path")
    if yaml_path:
        with open(yaml_path, "r") as f:
            yaml_config = yaml.safe_load(f)

    if not yaml_config or "model-car" not in yaml_config:
        return

    model_car_data = yaml_config["model-car"]
    default_serving_config = yaml_config.get("default", {})

    if not isinstance(model_car_data, list):
        raise ValueError("Invalid format for `model-car` in YAML. Expected a list of objects.")

    if not metafunc.cls:
        return

    params = []
    ids = []

    for model_car in model_car_data:
        if not model_car or not isinstance(model_car, dict):
            continue

        name = model_car.get("name", "").strip()
        image = model_car.get("image", "").strip()
        execution_mode = (
            model_car.get("execution_mode", "").strip()
            or default_serving_config.get("execution_mode", "sequential").strip()
        )

        if not name or not image:
            continue

        model_output_type = model_car.get("model_output_type", "text")
        serving_config = model_car.get("serving_arguments") or default_serving_config.get("serving_arguments", {})
        args = serving_config.get("args", [])
        gpu_count = serving_config.get("gpu_count", 1)

        if metafunc.cls.__name__ == "TestVLLMModelCarRaw":
            param, test_id = build_raw_params(
                name=name,
                image=image,
                args=args,
                gpu_count=gpu_count,
                execution_mode=execution_mode,
                model_output_type=model_output_type,
            )
        else:
            continue

        params.append(param)
        ids.append(test_id)

    if params:
        metafunc.parametrize(
            argnames=(
                "model_namespace, model_car_serving_runtime, vllm_model_car_inference_service, deployment_config"
            ),
            argvalues=params,
            indirect=True,
            ids=ids,
        )


@pytest.fixture
def vllm_model_car_pod_resource(admin_client: DynamicClient, vllm_model_car_inference_service: InferenceService) -> Pod:
    return get_pods_by_isvc_label(client=admin_client, isvc=vllm_model_car_inference_service)[0]
