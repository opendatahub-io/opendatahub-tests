import copy
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, cast

import pytest
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.template import Template
from pytest_testconfig import config as py_config
from simple_logger.logger import get_logger
from syrupy.extensions.json import JSONSnapshotExtension

from tests.model_serving.model_runtime.triton.basic_model_deployment.utils import (
    get_gpu_identifier,
    get_template_name,
)
from tests.model_serving.model_runtime.triton.constant import (
    PREDICT_RESOURCES,
    RUNTIME_MAP,
)
from utilities.constants import (
    KServeDeploymentType,
    Protocols,
)
from utilities.inference_utils import create_isvc
from utilities.infra import get_pods_by_isvc_label
from utilities.serving_runtime import ServingRuntimeFromTemplate

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="class")
def triton_grpc_serving_runtime_template(admin_client: DynamicClient, triton_runtime_image: str) -> Generator[Template]:
    with create_triton_template(
        admin_client=admin_client, protocol=Protocols.GRPC, triton_runtime_image=triton_runtime_image
    ) as template:
        yield template


@pytest.fixture(scope="class")
def triton_rest_serving_runtime_template(admin_client: DynamicClient, triton_runtime_image: str) -> Generator[Template]:
    with create_triton_template(
        admin_client=admin_client, protocol=Protocols.REST, triton_runtime_image=triton_runtime_image
    ) as template:
        yield template


@contextmanager
def create_triton_template(
    admin_client: DynamicClient, protocol: str, triton_runtime_image: str
) -> Generator[Template, Any, Any]:
    template_dict = {
        "apiVersion": "template.openshift.io/v1",
        "kind": "Template",
        "metadata": {
            "name": f"triton-{protocol}-runtime-template",
            "namespace": py_config["applications_namespace"],
        },
        "objects": [create_triton_serving_runtime(protocol=protocol, triton_runtime_image=triton_runtime_image)],
        "parameters": [],
    }

    with Template(
        client=admin_client,
        namespace=py_config["applications_namespace"],
        kind_dict=template_dict,
    ) as template:
        yield template


def create_triton_serving_runtime(protocol: str, triton_runtime_image: str) -> dict[str, Any]:
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

    kserve_container: list[dict[str, Any]] = [
        {
            "name": "kserve-container",
            "image": triton_runtime_image,
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

    supported_model_formats: list[dict[str, Any]] = [
        {"name": "tensorrt", "version": "8", "autoSelect": True, "priority": 1},
        {"name": "tensorflow", "version": "1", "autoSelect": True, "priority": 1},
        {"name": "tensorflow", "version": "2", "autoSelect": True, "priority": 1},
        {"name": "onnx", "version": "1", "autoSelect": True, "priority": 1},
        {"name": "pytorch", "version": "1", "autoSelect": True},
        {"name": "triton", "version": "2", "autoSelect": True, "priority": 1},
        {"name": "xgboost", "version": "1", "autoSelect": True},
        {"name": "python", "version": "1", "autoSelect": True},
    ]

    return {
        "apiVersion": "serving.kserve.io/v1alpha1",
        "kind": "ServingRuntime",
        "metadata": {
            "name": RUNTIME_MAP.get(protocol, "triton-runtime"),
            "annotations": {
                "prometheus.kserve.io/path": "/metrics",
                "prometheus.kserve.io/port": "8002",
            },
        },
        "spec": {
            "containers": kserve_container,
            "volumes": volumes,
            "protocolVersions": ["v2", "grpc-v2"],
            "supportedModelFormats": supported_model_formats,
        },
    }


@pytest.fixture(scope="class")
def triton_serving_runtime(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    protocol: str,
    supported_accelerator_type: str | None,
) -> Generator[ServingRuntime]:
    template_name = get_template_name(protocol=protocol, accelerator_type=supported_accelerator_type)
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=RUNTIME_MAP.get(protocol, "triton-runtime"),
        namespace=model_namespace.name,
        template_name=template_name,
        deployment_type=request.param.get("deployment_type", KServeDeploymentType.RAW_DEPLOYMENT),
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def triton_inference_service(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    triton_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    triton_model_service_account: ServiceAccount,
    supported_accelerator_type: str | None,
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
        identifier = get_gpu_identifier(accelerator_type=supported_accelerator_type)
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
def triton_model_service_account(admin_client: DynamicClient, kserve_s3_secret: Secret) -> Generator[ServiceAccount]:
    with ServiceAccount(
        client=admin_client,
        namespace=kserve_s3_secret.namespace,
        name="triton-models-bucket-sa",
        secrets=[{"name": kserve_s3_secret.name}],
    ) as sa:
        yield sa


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
        raise ResourceNotFoundError(f"No pods found for InferenceService {triton_inference_service.name}")
    return pods[0]
