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
from tests.model_serving.model_runtime.model_validation.utils import kserve_registry_pull_secret
from tests.model_serving.model_runtime.model_validation.constant import PULL_SECRET_NAME
from pytest import FixtureRequest
from kubernetes.dynamic import DynamicClient
from simple_logger.logger import get_logger
from utilities.serving_runtime import ServingRuntimeFromTemplate
from utilities.inference_utils import create_isvc
from utilities.infra import create_ns
from tests.model_serving.model_runtime.vllm.utils import validate_supported_quantization_schema
from tests.model_serving.model_runtime.model_validation.constant import (
    INFERENCE_SERVICE_PORT,
    CONTAINER_PORT,
    TIMEOUT_20MIN,
)
from tests.model_serving.model_runtime.model_validation.utils import safe_k8s_name
from tests.model_serving.model_runtime.model_validation.constant import (
    BASE_SEVERRLESS_DEPLOYMENT_CONFIG,
    BASE_RAW_DEPLOYMENT_CONFIG,
)
import yaml
from syrupy.extensions.json import JSONSnapshotExtension

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="class")
def vllm_model_car_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    dynamic_model_namespace: Namespace,
    modelcar_serving_runtime: ServingRuntime,
    supported_accelerator_type: str,
    modelcar_image_uri: str,
    registry_pull_secret: str,
    registry_host: str,
    deployment_config: dict[str, Any],
) -> Generator[InferenceService, Any, Any]:
    deployment_type = deployment_config.get("deployment_type", KServeDeploymentType.SERVERLESS).lower()
    name = safe_k8s_name(name=f"{modelcar_image_uri}-{deployment_type}", max_len=20)

    # Dynamically create pull secret in the correct namespace
    with kserve_registry_pull_secret(
        admin_client=admin_client,
        name=PULL_SECRET_NAME,
        namespace=dynamic_model_namespace.name,
        registry_pull_secret=registry_pull_secret,
        registry_host=registry_host,
    ):
        isvc_kwargs = {
            "client": admin_client,
            "name": name,
            "namespace": dynamic_model_namespace.name,
            "runtime": modelcar_serving_runtime.name,
            "storage_uri": modelcar_image_uri,
            "model_format": modelcar_serving_runtime.instance.spec.supportedModelFormats[0].name,
            "deployment_mode": deployment_config.get("deployment_type", KServeDeploymentType.SERVERLESS),
            "image_pull_secrets": [PULL_SECRET_NAME],
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
def modelcar_serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    dynamic_model_namespace: Namespace,
    supported_accelerator_type: str,
    vllm_runtime_image: str,
) -> Generator[ServingRuntime, None, None]:
    accelerator_type = supported_accelerator_type.lower()
    template_name = TEMPLATE_MAP.get(accelerator_type, RuntimeTemplates.VLLM_CUDA)
    print(f"using template: {template_name}")
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="vllm-runtime",
        namespace=dynamic_model_namespace.name,
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


@pytest.fixture(scope="function")
def modelcar_image_uri(request: FixtureRequest, model_image_name: str | list[str], registry_host: str) -> str:
    """
    Returns the model image URI for the modelcar image.
    If the model image name is not provided, it skips the test.
    """
    param_index = request.node.callspec.indices.get("modelcar_image_uri", None)
    if isinstance(model_image_name, list):
        if param_index is not None and param_index < len(model_image_name):
            override = model_image_name[param_index]
        else:
            pytest.skip("model_image_name completed, no more parameters to test.")
    else:
        override = model_image_name
    return f"oci://{registry_host}/rhelai1/{override}"


@pytest.fixture
def response_snapshot(snapshot: Any) -> Any:
    return snapshot.use_extension(extension_class=JSONSnapshotExtension)


@pytest.fixture(scope="class")
def dynamic_model_namespace(
    request: FixtureRequest,
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    teardown_resources: bool,
    deployment_config: dict[str, Any],
) -> Generator[Namespace, Any, Any]:
    if request.param.get("modelmesh-enabled"):
        request.getfixturevalue(argname="enabled_modelmesh_in_dsc")

    deployment_type = deployment_config.get("deployment_type", KServeDeploymentType.SERVERLESS)
    deployment_key = deployment_type.lower()  # 'rawdeployment' or 'serverless'
    dynamic_name = f"{deployment_key}-models-ns"

    ns = Namespace(client=admin_client, name=dynamic_name)
    LOGGER.info(f"Creating shared namespace: {ns.name}")

    if pytestconfig.option.post_upgrade:
        yield ns
        ns.clean_up()
    else:
        with create_ns(
            client=admin_client,
            name=dynamic_name,
            pytest_request=request,
            teardown=teardown_resources,
        ) as ns:
            yield ns


@pytest.fixture(scope="class")
def deployment_config(
    request: FixtureRequest,
    serving_argument: list[str],
) -> dict[str, Any]:
    """
    Fixture to provide the base deployment configuration for serverless deployments.
    """
    deployment_type = request.param.get("deployment_type", KServeDeploymentType.SERVERLESS)

    config = (
        BASE_SEVERRLESS_DEPLOYMENT_CONFIG.copy()
        if deployment_type == KServeDeploymentType.SERVERLESS
        else BASE_RAW_DEPLOYMENT_CONFIG.copy()
    )
    config["runtime_argument"] = serving_argument
    config["deployment_type"] = deployment_type
    return config


def build_raw_params(image_list: list[str]) -> tuple[list[pytest.param], list[str]]:
    params = []
    ids = []
    for image in image_list:
        image = image.strip()
        if not image:
            continue
        ids.append(f"{image}-raw")
        params.append(
            pytest.param(
                {"deployment_type": KServeDeploymentType.RAW_DEPLOYMENT, "modelmesh-enabled": True},
                {"deployment_type": KServeDeploymentType.RAW_DEPLOYMENT},
                {
                    "modelcar_image_uri": image,
                    "gpu_count": 1,
                    "timeout": TIMEOUT_20MIN,
                },
                {"deployment_type": KServeDeploymentType.RAW_DEPLOYMENT},
                image,
                id=f"{image}-raw",
                marks=[pytest.mark.rawdeployment],
            )
        )
    return params, ids


def build_serverless_params(image_list: list[str]) -> tuple[list[pytest.param], list[str]]:
    params = []
    ids = []
    for image in image_list:
        image = image.strip()
        if not image:
            continue
        ids.append(f"{image}-serverless")
        params.append(
            pytest.param(
                {"deployment_type": KServeDeploymentType.SERVERLESS, "modelmesh-enabled": False},
                {"deployment_type": KServeDeploymentType.SERVERLESS},
                {
                    "modelcar_image_uri": image,
                    "gpu_count": 1,
                    "timeout": TIMEOUT_20MIN,
                },
                {"deployment_type": KServeDeploymentType.SERVERLESS},
                image,
                id=f"{image}-serverless",
                marks=[pytest.mark.serverless],
            )
        )
    return params, ids


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    yaml_config = None
    yaml_path = metafunc.config.getoption(name="modelcar_yaml_path")
    if yaml_path:
        with open(yaml_path, "r") as f:
            yaml_config = yaml.safe_load(f)

    if yaml_config and "modelcar_image_name" in yaml_config:
        model_images = yaml_config["modelcar_image_name"]
        if isinstance(model_images, str):
            image_list = [x.strip() for x in model_images.split(",")]
        elif isinstance(model_images, list):
            image_list = model_images
        else:
            raise ValueError("Invalid format for `model_image_name` in YAML.")
    else:
        image_arg = metafunc.config.getoption(name="modelcar_image_name")
        image_list = image_arg.split(",") if image_arg else []

    if metafunc.cls.__name__ == "TestVLLMModelcarRaw":
        params, ids = build_raw_params(image_list=image_list)
    elif metafunc.cls.__name__ == "TestVLLMModelcarServerless":
        params, ids = build_serverless_params(image_list)
    else:
        return

    metafunc.parametrize(
        argnames=(
            "dynamic_model_namespace, modelcar_serving_runtime, "
            "vllm_model_car_inference_service, deployment_config, modelcar_image_uri"
        ),
        argvalues=params,
        indirect=True,
        ids=ids,
    )
