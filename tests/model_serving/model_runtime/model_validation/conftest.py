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
)

import hashlib
import re
from syrupy.extensions.json import JSONSnapshotExtension

LOGGER = get_logger(name=__name__)


def safe_k8s_name(name: str, max_len: int = 20) -> str:
    """
    Convert a model image URI or name into a safe, human-readable Kubernetes name.
    Prioritizes model family and size, e.g., 'granite-8b-oci'.
    Falls back to hashed suffix if over max_len.
    """
    # Extract the base model name (e.g., "modelcar-granite-3-1-8b-base-quantized-w4a16")
    base = name.split("/")[-1].split(":")[0]  # Strip registry and tag

    # Clean and tokenize
    parts = base.replace("modelcar-", "").split("-")

    # Try to extract model family and size
    family = parts[0] if parts else "model"
    size = next((p for p in parts if re.match(r"^\d+b$", p)), "unk")

    readable = f"{family}-{size}-oci"

    if len(readable) <= max_len:
        return readable

    # Truncate with hash fallback
    hash_suffix = hashlib.sha1(readable.encode()).hexdigest()[:6]
    max_base_len = max_len - len("-" + hash_suffix)
    return f"{readable[:max_base_len]}-{hash_suffix}"


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
) -> Generator[InferenceService, Any, Any]:
    name = safe_k8s_name(name=modelcar_image_uri, max_len=20)

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
            "deployment_mode": request.param.get("deployment_mode", KServeDeploymentType.SERVERLESS),
            "image_pull_secrets": [registry_pull_secret],  # secret name
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


@pytest.fixture(scope="session")
def modelcar_image_uri(request: FixtureRequest) -> str | None:
    model_image_name = request.param
    if not model_image_name:
        return None
    return f"oci://registry.redhat.io/rhelai1/{model_image_name}"


@pytest.fixture
def response_snapshot(snapshot: Any) -> Any:
    return snapshot.use_extension(extension_class=JSONSnapshotExtension)


@pytest.fixture(scope="class")
def dynamic_model_namespace(
    request: FixtureRequest,
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    teardown_resources: bool,
    modelcar_image_uri: str,
) -> Generator[Namespace, Any, Any]:
    if request.param.get("modelmesh-enabled"):
        request.getfixturevalue(argname="enabled_modelmesh_in_dsc")

    name = safe_k8s_name(name=modelcar_image_uri, max_len=20)

    dynamic_name = f"{name}-ns"

    ns = Namespace(client=admin_client, name=dynamic_name)

    LOGGER.info(f"Creating dynamic namespace: {ns.name}")

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
