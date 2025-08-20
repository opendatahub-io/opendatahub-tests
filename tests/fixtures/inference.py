from typing import Generator, Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.serving_runtime import ServingRuntime

from utilities.constants import RuntimeTemplates, KServeDeploymentType
from utilities.inference_utils import create_isvc
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="class")
def vllm_cpu_runtime(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    port = 8032
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
                "args": [
                    f"--port={str(port)}",
                    "--model=/mnt/models",
                    "--served-model-name={{.Name}}",
                    "--dtype=float16",
                    "--enforce-eager",
                ],
                "env": [
                    {"name": "HF_HOME", "value": "/tmp/hf_home"}  # ✅ Add HF_HOME env var
                ],
                "command": ["python", "-m", "vllm.entrypoints.openai.api_server"],
                "ports": [{"containerPort": port, "protocol": "TCP"}],
                "volumeMounts": [{"mountPath": "/dev/shm", "name": "shm"}],
            }
        },
        volumes=[{"emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"}, "name": "shm"}],
    ) as serving_runtime:
        yield serving_runtime


@pytest.fixture(scope="class")
def qwen25_05B_instruct(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    vllm_cpu_runtime: ServingRuntime,
) -> Generator[InferenceService, Any, Any]:
    model_name = "qwen"
    with create_isvc(
        client=admin_client,
        name=model_name,
        namespace=model_namespace.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        model_format="vLLM",
        runtime=vllm_cpu_runtime.name,
        storage_uri="oci://quay.io/redhat-ai-services/modelcar-catalog"
        "@sha256:b96bd7aac6d2d6d0ec86166037e68fdff31152588846ffbd9e8fc41dd571b9e8",
        wait_for_predictor_pods=False,
        resources={
            "requests": {"cpu": "1", "memory": "8Gi"},
            "limits": {"cpu": "2", "memory": "10Gi"},
        },
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def qwen25_05B_instruct_url(qwen25_05B_instruct: InferenceService) -> str:
    return f"http://{qwen25_05B_instruct.name}-predictor.{qwen25_05B_instruct.namespace}.svc.cluster.local:8032/v1"
