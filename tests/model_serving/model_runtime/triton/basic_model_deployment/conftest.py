from typing import cast, Any, Generator, List, Dict
import copy
import pytest
from contextlib import contextmanager
from syrupy.extensions.json import JSONSnapshotExtension
from pytest_testconfig import config as py_config

from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.template import Template
from ocp_resources.service_account import ServiceAccount

from tests.model_serving.model_runtime.triton.constant import (
    PREDICT_RESOURCES,
    RUNTIME_MAP,
    TRITON_SERVER_IMAGE,
)
from tests.model_serving.model_runtime.triton.basic_model_deployment.utils import (
    kserve_s3_endpoint_secret,
    get_template_name,
    get_gpu_identifier,
)

from utilities.constants import (
    KServeDeploymentType,
    Protocols,
)
from utilities.inference_utils import create_isvc
from utilities.infra import get_pods_by_isvc_label
from utilities.serving_runtime import ServingRuntimeFromTemplate

from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="session")
def root_dir(pytestconfig: pytest.Config) -> Any:
    return pytestconfig.rootpath


@pytest.fixture(scope="session")
def supported_accelerator_type(pytestconfig: pytest.Config) -> str:
    return py_config.get("accelerator_type", "nvidia")


@pytest.fixture(scope="class")
def protocol(request: pytest.FixtureRequest) -> str:
    return request.param["protocol_type"]


@pytest.fixture(scope="class")
def triton_grpc_serving_runtime_template(admin_client: DynamicClient) -> Generator[Template, None, None]:
    with create_triton_template(admin_client=admin_client, protocol=Protocols.GRPC) as template:
        yield template


@pytest.fixture(scope="class")
def triton_rest_serving_runtime_template(admin_client: DynamicClient) -> Generator[Template, None, None]:
    with create_triton_template(admin_client=admin_client, protocol=Protocols.REST) as template:
        yield template


@contextmanager
def create_triton_template(admin_client: DynamicClient, protocol: str) -> Generator[Template, Any, Any]:
    with Template(
        client=admin_client,
        name=f"triton-{protocol}-runtime-template",
        namespace=py_config["applications_namespace"],
        objects=[create_triton_serving_runtime(protocol=protocol)],
    ) as template:
        yield template


def create_triton_serving_runtime(protocol: str) -> ServingRuntime:
    volumes = []
    volume_mounts = []
    if protocol == Protocols.GRPC:
        volumes.append({"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"}})
        volume_mounts.append({"name": "shm", "mountPath": "/dev/shm"})

    port_config = {
        "name": "h2c" if protocol == Protocols.GRPC else "http1",
        "containerPort": 9000 if protocol == Protocols.GRPC else 8080,
        "protocol": "TCP",
    }

    container_args = [
        "tritonserver",
        "--model-store=/mnt/models",
        f"--{'grpc' if protocol == Protocols.GRPC else 'http'}-port={port_config['containerPort']}",
        f"--{'allow-grpc' if protocol == Protocols.GRPC else 'allow-http'}=True",
    ]

    kserve_container: List[Dict[str, Any]] = [
        {
            "name": "kserve-container",
            "image": TRITON_SERVER_IMAGE,
            "args": container_args,
            "ports": [port_config],
            "volumeMounts": volume_mounts,
            "resources": {
                "requests": {
                    "cpu": "1",
                    "memory": "2Gi",
                },
                "limits": {
                    "cpu": "1",
                    "memory": "2Gi",
                },
            },
        }
    ]

    supported_model_formats: List[Dict[str, Any]] = [
        {"name": "tensorrt", "version": "8", "autoSelect": True, "priority": 1},
        {"name": "tensorflow", "version": "1", "autoSelect": True, "priority": 1},
        {"name": "tensorflow", "version": "2", "autoSelect": True, "priority": 1},
        {"name": "onnx", "version": "1", "autoSelect": True, "priority": 1},
        {"name": "pytorch", "version": "1", "autoSelect": True},
        {"name": "triton", "version": "2", "autoSelect": True, "priority": 1},
        {"name": "xgboost", "version": "1", "autoSelect": True},
        {"name": "python", "version": "1", "autoSelect": True},
    ]

    return ServingRuntime(
        name=f"triton-kserve-{protocol}",
        spec_annotations={
            "prometheus.kserve.io/path": "/metrics",
            "prometheus.kserve.io/port": "8002",
        },
        containers=kserve_container,
        volumes=volumes,
        protocol_versions=["v2", "grpc-v2"],
        supported_model_formats=supported_model_formats,
    )


@pytest.fixture(scope="function")
def triton_serving_runtime(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    triton_runtime_image: str,
    protocol: str,
    supported_accelerator_type: str,
) -> Generator[ServingRuntime, None, None]:
    template_name = get_template_name(protocol, supported_accelerator_type)
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=RUNTIME_MAP.get(protocol, "triton-runtime"),
        namespace=model_namespace.name,
        template_name=template_name,
        deployment_type=request.param["deployment_type"],
        runtime_image=triton_runtime_image,
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="function")
def triton_inference_service(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    triton_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    triton_model_service_account: ServiceAccount,
    supported_accelerator_type: str,
) -> Generator[InferenceService, Any, Any]:
    params = request.param
    model_format = params.get(
        "model_format",
        triton_serving_runtime.instance.spec.supportedModelFormats[0].name,
    )
    gpu_count = params.get("gpu_count", 0)
    timeout = params.get("timeout")
    min_replicas = params.get("min-replicas")
    service_config = {
        "client": admin_client,
        "name": params.get("name"),
        "namespace": model_namespace.name,
        "runtime": triton_serving_runtime.name,
        "storage_uri": s3_models_storage_uri,
        "model_format": model_format,
        "model_service_account": triton_model_service_account.name,
        "deployment_mode": params.get("deployment_type", KServeDeploymentType.RAW_DEPLOYMENT),
        "external_route": params.get("enable_external_route", False),
    }
    resources = copy.deepcopy(cast(dict[str, dict[str, str]], PREDICT_RESOURCES["resources"]))
    if gpu_count > 0:
        identifier = get_gpu_identifier(supported_accelerator_type)
        resources["requests"][identifier] = gpu_count
        resources["limits"][identifier] = gpu_count

        if gpu_count > 1:
            service_config["volumes"] = PREDICT_RESOURCES["volumes"]
            service_config["volumes_mounts"] = PREDICT_RESOURCES["volume_mounts"]
    service_config["resources"] = resources

    if timeout:
        service_config["timeout"] = timeout
    if min_replicas:
        service_config["min_replicas"] = min_replicas

    with create_isvc(**service_config) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def triton_model_service_account(
    admin_client: DynamicClient, kserve_s3_secret: Secret
) -> Generator[ServiceAccount, None, None]:
    with ServiceAccount(
        client=admin_client,
        namespace=kserve_s3_secret.namespace,
        name="triton-models-bucket-sa",
        secrets=[{"name": kserve_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def kserve_s3_secret(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
) -> Generator[Secret, None, None]:
    with kserve_s3_endpoint_secret(
        admin_client=admin_client,
        name="triton-models-bucket-secret",
        namespace=model_namespace.name,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_region=models_s3_bucket_region,
        aws_s3_endpoint=models_s3_bucket_endpoint,
    ) as secret:
        yield secret


@pytest.fixture
def triton_response_snapshot(snapshot: Any) -> Any:
    return snapshot.use_extension(extension_class=JSONSnapshotExtension)


@pytest.fixture
def triton_pod_resource(
    admin_client: DynamicClient,
    triton_inference_service: InferenceService,
) -> Pod:
    pods = get_pods_by_isvc_label(client=admin_client, isvc=triton_inference_service)
    if not pods:
        raise RuntimeError(f"No pods found for InferenceService {triton_inference_service.name}")
    return pods[0]
