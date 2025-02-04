from typing import Any, Generator
import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from tests.model_serving.model_runtime.vllm.utils import kserve_s3_endpoint_secret
from utilities.constants import KServeDeploymentType
from pytest import FixtureRequest
from syrupy.extensions.json import JSONSnapshotExtension
from tests.model_serving.model_runtime.vllm.utils import get_runtime_manifest
from tests.model_serving.model_server.utils import create_isvc
from tests.model_serving.model_runtime.vllm.constant import TEMPLATE_MAP, ACCELERATOR_IDENTIFIER, PREDICT_RESOURCES
from simple_logger.logger import get_logger
from utilities.infra import get_pods_by_isvc_label

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="class")
def serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    supported_accelerator_type: str,
    vllm_runtime_image: str,
) -> Generator[ServingRuntime, None, None]:
    accelerator_type = supported_accelerator_type.lower()
    template_name = TEMPLATE_MAP.get(accelerator_type, "vllm-runtime-template")
    manifest = get_runtime_manifest(
        client=admin_client,
        template_name=template_name,
        deployment_type=request.param["deployment_type"],
        runtime_image=vllm_runtime_image,
    )
    manifest["metadata"]["name"] = "vllm-runtime"
    manifest["metadata"]["namespace"] = model_namespace.name
    with ServingRuntime(client=admin_client, kind_dict=manifest) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="session")
def skip_if_no_supported_accelerator_type(supported_accelerator_type: str) -> None:
    if not supported_accelerator_type:
        pytest.skip("Accelartor type is not provide,vLLM test can not be run on CPU")


@pytest.fixture(scope="class")
def vllm_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    serving_runtime: ServingRuntime,
    supported_accelerator_type: str,
    s3_models_storage_uri: str,
    vllm_model_service_account: ServiceAccount,
) -> Generator[InferenceService, Any, Any]:
    isvc_kwargs = {
        "client": admin_client,
        "name": request.param["name"],
        "namespace": model_namespace.name,
        "runtime": serving_runtime.name,
        "storage_uri": s3_models_storage_uri,
        "model_format": serving_runtime.instance.spec.supportedModelFormats[0].name,
        "model_service_account": vllm_model_service_account.name,
        "deployment_mode": request.param.get("deployment_mode", KServeDeploymentType.SERVERLESS),
    }
    accelerator_type = supported_accelerator_type.lower()
    gpu_count = request.param.get("gpu_count")
    identifier = ACCELERATOR_IDENTIFIER.get(accelerator_type, "nvidia.com/gpu")
    resources: Any = PREDICT_RESOURCES["resources"]
    resources["requests"][identifier] = gpu_count
    resources["limits"][identifier] = gpu_count
    isvc_kwargs["resources"] = resources

    if gpu_count > 1:
        isvc_kwargs["volumes"] = PREDICT_RESOURCES["volumes"]
        isvc_kwargs["volumes_mounts"] = PREDICT_RESOURCES["volume_mounts"]
    if arguments := request.param.get("runtime_argument"):
        arguments = [arg for arg in arguments if not arg.startswith("--tensor-parallel-size")]
        arguments.append(f"--tensor-parallel-size={gpu_count}")
        isvc_kwargs["argument"] = arguments

    if min_replicas := request.param.get("min-replicas"):
        isvc_kwargs["min_replicas"] = min_replicas

    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def vllm_model_service_account(admin_client: DynamicClient, kserve_endpoint_s3_secret: Secret) -> ServiceAccount:
    with ServiceAccount(
        client=admin_client,
        namespace=kserve_endpoint_s3_secret.namespace,
        name="models-bucket-sa",
        secrets=[{"name": kserve_endpoint_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def kserve_endpoint_s3_secret(
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


@pytest.fixture
def response_snapshot(snapshot: Any) -> Any:
    return snapshot.use_extension(extension_class=JSONSnapshotExtension)


@pytest.fixture
def get_pod_name_resource(admin_client: DynamicClient, vllm_inference_service: InferenceService) -> Pod:
    return get_pods_by_isvc_label(client=admin_client, isvc=vllm_inference_service)[0]
