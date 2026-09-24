from collections.abc import Generator
from typing import Any

import pytest
from pytest import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.serving_runtime import ServingRuntime

from tests.ai_safety.image_constants import AiSafetyImages
from tests.model_serving.image_constants import ModelServingImages
from tests.ai_safety.trustyai_service.trustyai_service_utils import (
    wait_for_isvc_deployment_registered_by_trustyai_service,
)
from utilities.constants import KServeDeploymentType, ModelFormat, RuntimeTemplates
from utilities.inference_utils import create_isvc
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="class")
def ovms_runtime(admin_client: DynamicClient, model_namespace: Namespace) -> Generator[ServingRuntime, Any, Any]:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=f"{ModelFormat.OVMS}-1.x",
        namespace=model_namespace.name,
        template_name=RuntimeTemplates.OVMS_KSERVE,
        multi_model=False,
        enable_http=False,
        enable_grpc=True,
        model_format_name={"name": ModelFormat.ONNX, "version": "1"},
    ) as sr:
        yield sr


@pytest.fixture(scope="class")
def onnx_loan_model(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    request: FixtureRequest,
    cluster_architecture: str,
    kserve_raw_config: ConfigMap,
    kserve_logger_ca_bundle: ConfigMap,
) -> Generator[InferenceService, Any, Any]:
    if cluster_architecture == "s390x":
        runtime = request.getfixturevalue("triton_runtime")
        storage_uri = AiSafetyImages.LOAN_MODEL_ALPHA_ONNXMLIR
        model_format = ModelFormat.ONNX_MLIR
    else:
        runtime = request.getfixturevalue("ovms_runtime")
        storage_uri = AiSafetyImages.LOAN_MODEL_ALPHA
        model_format = ModelFormat.ONNX
    with create_isvc(
        client=admin_client,
        name="demo-loan-nn-onnx-alpha",
        namespace=model_namespace.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        model_format=model_format,
        runtime=runtime.name,
        storage_uri=storage_uri,
        min_replicas=1,
        resources={"limits": {"cpu": "2", "memory": "8Gi"}, "requests": {"cpu": "1", "memory": "4Gi"}},
        enable_auth=True,
        external_route=True,
        model_version="1",
        wait=True,
        wait_for_predictor_pods=False,
    ) as isvc:
        wait_for_isvc_deployment_registered_by_trustyai_service(
            client=admin_client,
            isvc=isvc,
            runtime_name=runtime.name,
        )
        yield isvc


@pytest.fixture(scope="class")
def triton_runtime(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    runtime_dict = {
        "apiVersion": "serving.kserve.io/v1alpha1",
        "kind": "ServingRuntime",
        "metadata": {
            "name": "ibmz-triton-rest",
            "namespace": model_namespace.name,
            "labels": {
                "opendatahub.io/dashboard": "true",
            },
        },
        "spec": {
            "containers": [
                {
                    "name": "kserve-container",
                    "command": [
                        "/bin/sh",
                        "-c",
                    ],
                    "args": [
                        "/opt/tritonserver/bin/tritonserver "
                        "--model-repository=/mnt/models "
                        "--http-port=8000 "
                        "--grpc-port=8001 "
                        "--metrics-port=8002"
                    ],
                    "image": ModelServingImages.TRITON_S390X,
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                        "capabilities": {
                            "drop": ["ALL"],
                        },
                        "runAsNonRoot": True,
                        "seccompProfile": {
                            "type": "RuntimeDefault",
                        },
                    },
                    "resources": {
                        "limits": {
                            "cpu": "2",
                            "memory": "4Gi",
                        },
                        "requests": {
                            "cpu": "2",
                            "memory": "4Gi",
                        },
                    },
                    "ports": [
                        {
                            "containerPort": 8000,
                            "protocol": "TCP",
                        },
                    ],
                }
            ],
            "protocolVersions": [
                "v2",
                "grpc-v2",
            ],
            "supportedModelFormats": [
                {
                    "name": "onnx-mlir",
                    "version": "1",
                    "autoSelect": True,
                },
            ],
        },
    }

    with ServingRuntime(
        client=admin_client,
        kind_dict=runtime_dict,
        teardown=True,
    ) as runtime:
        yield runtime
