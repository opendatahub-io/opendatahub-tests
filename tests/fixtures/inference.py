from typing import Generator, Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.serving_runtime import ServingRuntime

from utilities.constants import (
    RuntimeTemplates,
    KServeDeploymentType,
    QWEN_MODEL_NAME,
    LLMdInferenceSimConfig,
)
from utilities.inference_utils import create_isvc
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="class")
def vllm_cpu_runtime(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    minio_pod: Pod,
    minio_service: Service,
    minio_data_connection: Secret,
) -> Generator[ServingRuntime, Any, Any]:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="vllm-runtime-cpu-fp16",
        namespace=model_namespace.name,
        template_name=RuntimeTemplates.VLLM_CUDA,
        deployment_type=KServeDeploymentType.RAW_DEPLOYMENT,
        runtime_image="quay.io/rh-aiservices-bu/vllm-cpu-openai-ubi9"
        "@sha256:ada6b3ba98829eb81ae4f89364d9b431c0222671eafb9a04aa16f31628536af2",
        containers={
            "kserve-container": {
                "args": ["--port=8032", "--model=/mnt/models", "--served-model-name={{.Name}}"],
                "ports": [{"containerPort": 8032, "protocol": "TCP"}],
                "volumeMounts": [{"mountPath": "/dev/shm", "name": "shm"}],
            }
        },
        volumes=[{"emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"}, "name": "shm"}],
    ) as serving_runtime:
        yield serving_runtime


@pytest.fixture(scope="class")
def qwen_isvc(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    minio_pod: Pod,
    minio_service: Service,
    minio_data_connection: Secret,
    vllm_cpu_runtime: ServingRuntime,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name=QWEN_MODEL_NAME,
        namespace=model_namespace.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        model_format="vLLM",
        runtime=vllm_cpu_runtime.name,
        storage_key=minio_data_connection.name,
        storage_path="Qwen2.5-0.5B-Instruct",
        wait_for_predictor_pods=False,
        resources={
            "requests": {"cpu": "2", "memory": "10Gi"},
            "limits": {"cpu": "2", "memory": "12Gi"},
        },
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def qwen_isvc_url(qwen_isvc: InferenceService) -> str:
    return f"http://{qwen_isvc.name}-predictor.{qwen_isvc.namespace}.svc.cluster.local:8032/v1"


@pytest.fixture(scope="class")
def llm_d_inference_sim_serving_runtime(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    with ServingRuntime(
        client=admin_client,
        name=LLMdInferenceSimConfig.serving_runtime_name,
        namespace=model_namespace.name,
        annotations={
            "description": "LLM-d Simulator KServe",
            "opendatahub.io/template-display-name": "LLM-d Inference Simulator Runtime",
            "openshift.io/display-name": "LLM-d Inference Simulator Runtime",
            "serving.kserve.io/enable-agent": "false",
        },
        label={
            "app.kubernetes.io/component": LLMdInferenceSimConfig.name,
            "app.kubernetes.io/instance": "llm-d-inference-sim-kserve",
            "app.kubernetes.io/name": "llm-d-sim",
            "app.kubernetes.io/version": "1.0.0",
            "opendatahub.io/dashboard": "true",
        },
        spec_annotations={
            "prometheus.io/path": "/metrics",
            "prometheus.io/port": "8000",
        },
        spec_labels={
            "opendatahub.io/dashboard": "true",
        },
        containers=[
            {
                "name": "kserve-container",
                "image": "quay.io/trustyai_testing/llmd-inference-sim-dataset-builtin"
                "@sha256:dfaa32cf0878a2fb522133e34369412c90e8ffbfa18b690b92602cf7c019fbbe",
                "imagePullPolicy": "Always",
                "args": ["--model", LLMdInferenceSimConfig.model_name, "--port", str(LLMdInferenceSimConfig.port)],
                "ports": [{"containerPort": LLMdInferenceSimConfig.port, "protocol": "TCP"}],
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                },
                "livenessProbe": {
                    "failureThreshold": 3,
                    "httpGet": {"path": "/health", "port": LLMdInferenceSimConfig.port, "scheme": "HTTP"},
                    "initialDelaySeconds": 15,
                    "periodSeconds": 20,
                    "timeoutSeconds": 5,
                },
                "readinessProbe": {
                    "failureThreshold": 3,
                    "httpGet": {"path": "/health", "port": LLMdInferenceSimConfig.port, "scheme": "HTTP"},
                    "initialDelaySeconds": 5,
                    "periodSeconds": 10,
                    "timeoutSeconds": 5,
                },
            }
        ],
        multi_model=False,
        supported_model_formats=[{"autoSelect": True, "name": LLMdInferenceSimConfig.name}],
    ) as serving_runtime:
        yield serving_runtime


@pytest.fixture(scope="class")
def llm_d_inference_sim_isvc(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    llm_d_inference_sim_serving_runtime: ServingRuntime,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name=LLMdInferenceSimConfig.isvc_name,
        namespace=model_namespace.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        model_format=LLMdInferenceSimConfig.name,
        runtime=llm_d_inference_sim_serving_runtime.name,
        wait_for_predictor_pods=True,
        min_replicas=1,
        max_replicas=1,
        resources={
            "requests": {"cpu": "1", "memory": "1Gi"},
            "limits": {"cpu": "1", "memory": "1Gi"},
        },
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def llm_d_inference_sim_isvc_url(llm_d_inference_sim_isvc: InferenceService) -> str:
    return (
        f"http://{llm_d_inference_sim_isvc.name}-predictor."
        f"{llm_d_inference_sim_isvc.namespace}.svc.cluster.local:{LLMdInferenceSimConfig.port}/v1"
    )
