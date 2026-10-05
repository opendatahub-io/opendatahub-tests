from collections.abc import Generator
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.template import Template
from pytest import FixtureRequest
from pytest_testconfig import config as py_config

from tests.model_serving.model_runtime.triton.constant import (
    PREDICT_RESOURCES,
    RUNTIME_MAP,
)
from tests.model_serving.model_runtime.triton.S3.utils import (
    get_gpu_identifier,
)
from utilities.constants import KServeDeploymentType, Protocols
from utilities.general import download_model_data
from utilities.inference_utils import create_isvc
from utilities.infra import get_pods_by_isvc_label


@contextmanager
def create_triton_template(
    admin_client: DynamicClient, protocol: str, triton_runtime_image: str
) -> Generator[Template, Any, Any]:
    """Create a Triton ServingRuntime template dynamically."""
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
    """Create Triton ServingRuntime object definition."""
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
                "requests": {"cpu": "1", "memory": "2Gi"},
                "limits": {"cpu": "1", "memory": "2Gi"},
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
def triton_model_pvc(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    """PVC for storing Triton model data downloaded from S3."""
    pvc_kwargs: dict[str, Any] = {
        "name": "triton-model-pvc",
        "namespace": model_namespace.name,
        "client": admin_client,
        "size": request.param["pvc-size"],
        "accessmodes": request.param.get("access-modes", "ReadWriteOnce"),
    }
    if storage_class_name := request.param.get("storage-class-name"):
        pvc_kwargs["storage_class"] = storage_class_name

    with PersistentVolumeClaim(**pvc_kwargs) as pvc:
        yield pvc


@pytest.fixture(scope="class")
def triton_pvc_downloaded_model_data(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    triton_model_pvc: PersistentVolumeClaim,
    aws_secret_access_key: str,
    aws_access_key_id: str,
    models_s3_bucket_name: str,
    models_s3_bucket_endpoint: str,
    models_s3_bucket_region: str,
) -> str:
    """Download Triton model data from the models S3 bucket into the PVC.

    Note: GPU zone selection is not needed for CPU-only tests.
    If GPU tests are added in the future, consider adding node_selector
    logic to pin PVC download pods to GPU availability zones.
    """
    return download_model_data(
        client=admin_client,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        model_namespace=model_namespace.name,
        model_pvc_name=triton_model_pvc.name,
        bucket_name=models_s3_bucket_name,
        aws_endpoint_url=models_s3_bucket_endpoint,
        aws_default_region=models_s3_bucket_region,
        model_path=request.param["model-dir"],
        use_sub_path=True,
        restricted_scc_init=True,
    )


@pytest.fixture(scope="class")
def triton_pvc_serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    protocol: str,
    triton_runtime_image: str,
) -> Generator[ServingRuntime, Any, Any]:
    """Triton serving runtime for PVC-backed deployments using dynamic template."""
    # Create the runtime dict directly (no need for template extraction)
    runtime_dict = create_triton_serving_runtime(
        protocol=protocol,
        triton_runtime_image=triton_runtime_image,
    )
    # Set the namespace
    runtime_dict["metadata"]["namespace"] = model_namespace.name

    with ServingRuntime(
        client=admin_client,
        namespace=model_namespace.name,
        kind_dict=runtime_dict,
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def triton_pvc_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    triton_pvc_serving_runtime: ServingRuntime,
    supported_accelerator_type: str | None,
    triton_model_pvc: PersistentVolumeClaim,
    triton_pvc_downloaded_model_data: str,
) -> Generator[InferenceService, Any, Any]:
    """Triton InferenceService backed by PVC storage."""
    # Get model_format safely to avoid eager evaluation
    model_format = request.param.get("model_format")
    if not model_format:
        supported_formats = triton_pvc_serving_runtime.instance.spec.supportedModelFormats
        if supported_formats:
            model_format = supported_formats[0].name
        else:
            raise ValueError("No supported model formats found in serving runtime")

    isvc_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": request.param["name"],
        "namespace": model_namespace.name,
        "runtime": triton_pvc_serving_runtime.name,
        "storage_uri": f"pvc://{triton_model_pvc.name}/{triton_pvc_downloaded_model_data}",
        "model_format": model_format,
        "deployment_mode": request.param.get("deployment_mode", KServeDeploymentType.STANDARD),
        "external_route": request.param.get("enable_external_route", False),
    }

    gpu_count = request.param.get("gpu_count", 0)
    timeout = request.param.get("timeout")
    min_replicas = request.param.get("min-replicas")

    resources = deepcopy(x=PREDICT_RESOURCES["resources"])

    if gpu_count > 0:
        identifier = get_gpu_identifier(accelerator_type=supported_accelerator_type)
        resources["requests"][identifier] = gpu_count
        resources["limits"][identifier] = gpu_count

        if gpu_count > 1:
            isvc_kwargs["volumes"] = PREDICT_RESOURCES["volumes"]
            isvc_kwargs["volumes_mounts"] = PREDICT_RESOURCES["volume_mounts"]

    isvc_kwargs["resources"] = resources

    if timeout:
        isvc_kwargs["timeout"] = timeout

    if min_replicas:
        isvc_kwargs["min_replicas"] = min_replicas

    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc


@pytest.fixture
def triton_pod_resource(
    admin_client: DynamicClient,
    triton_pvc_inference_service: InferenceService,
) -> Pod:
    """Get the pod for the Triton InferenceService."""
    pods = get_pods_by_isvc_label(client=admin_client, isvc=triton_pvc_inference_service)
    if not pods:
        raise ResourceNotFoundError(f"No pods found for InferenceService {triton_pvc_inference_service.name}")
    return pods[0]


@pytest.fixture
def model_name(request: FixtureRequest) -> str:
    """Extract model_name from parametrize."""
    return request.param


@pytest.fixture
def input_path(request: FixtureRequest) -> str:
    """Extract input_path from parametrize."""
    return request.param
