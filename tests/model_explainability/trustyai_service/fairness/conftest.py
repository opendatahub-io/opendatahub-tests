from typing import Any, Generator

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime

from utilities.constants import ModelFormat, KServeDeploymentType
from utilities.infra import create_isvc


@pytest.fixture(scope="class")
def kserve_ovms_runtime(
    admin_client: DynamicClient, minio_data_connection: Secret, model_namespace: Namespace
) -> Generator[ServingRuntime, Any, Any]:
    supported_model_formats = [
        {"name": f"{ModelFormat.OPENVINO}_ir", "version": "opset13", "autoSelect": True},
        {"name": ModelFormat.ONNX, "version": "1"},
        {"name": ModelFormat.TENSORFLOW, "version": "1", "autoSelect": True},
        {"name": ModelFormat.TENSORFLOW, "version": "2", "autoSelect": True},
        {"name": "paddle", "version": "2", "autoSelect": True},
        {"name": "pytorch", "version": "2", "autoSelect": True},
    ]
    containers = [
        {
            "name": "kserve-container",
            "image": "quay.io/opendatahub/openvino_model_server:stable-nightly-2024-08-04",
            "args": [
                "--model_name={{.Name}}",
                "--port=8001",
                "--rest_port=8888",
                "--model_path=/mnt/models",
                "--file_system_poll_wait_seconds=0",
                "--grpc_bind_address=0.0.0.0",
                "--rest_bind_address=0.0.0.0",
                "--target_device=AUTO",
                "--metrics_enable",
            ],
            "ports": [
                {
                    "containerPort": 8888,
                    "protocol": "TCP",
                }
            ],
            "volumeMounts": [
                {
                    "mountPath": "/dev/shm",
                    "name": "shm",
                }
            ],
        }
    ]

    with ServingRuntime(
        client=admin_client,
        name=f"{ModelFormat.OVMS}-1.x",
        namespace=model_namespace.name,
        containers=containers,
        supported_model_formats=supported_model_formats,
        multi_model=False,
        protocol_versions=["v2", "grpc-v2"],
        annotations={
            "opendatahub.io/accelerator-name": "",
            "opendatahub.io/apiProtocol": "REST",
            "opendatahub.io/recommended-accelerators": '["nvidia.com/gpu"]',
            "opendatahub.io/template-display-name": "OpenVINO Model Server",
            "opendatahub.io/template-name": "kserve-ovms",
            "openshift.io/display-name": "ovms-1.x",
            "prometheus.io/path": "/metrics",
            "prometheus.io/port": "8888",
        },
        label={"opendatahub.io/dashboard": "true"},
        volumes=[{"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"}}],
    ) as sr:
        yield sr


@pytest.fixture(scope="class")
def onnx_loan_model(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    minio_data_connection: Secret,
    kserve_ovms_runtime: ServingRuntime,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name="demo-loan-nn-onnx-alpha",
        namespace=model_namespace.name,
        deployment_mode=KServeDeploymentType.SERVERLESS,
        model_format=ModelFormat.ONNX,
        runtime=kserve_ovms_runtime.name,
        storage_key=minio_data_connection.name,
        storage_path="ovms/loan_model_alpha",
        min_replicas=1,
        resources={"limits": {"cpu": "2", "memory": "8Gi"}, "requests": {"cpu": "1", "memory": "4Gi"}},
        enable_auth=True,
        model_version="1",
        wait=True,
        wait_for_predictor_pods=False,
    ) as inference_service:
        yield inference_service
