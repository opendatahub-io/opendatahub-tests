from typing import Generator, Any

import pytest
import yaml
from kubernetes.dynamic import DynamicClient
from llama_stack_client import LlamaStackClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.guardrails_orchestrator import GuardrailsOrchestrator
from ocp_resources.inference_service import InferenceService
from ocp_resources.llama_stack_distribution import LlamaStackDistribution
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.serving_runtime import ServingRuntime

from tests.model_explainability.guardrails.test_guardrails import MNT_MODELS
from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import (
    KServeDeploymentType,
    Labels,
    Ports,
    RuntimeTemplates,
)
from utilities.inference_utils import create_isvc
from utilities.serving_runtime import ServingRuntimeFromTemplate

ORCHESTRATOR_CONFIGMAP_NAME = "fms-orchestr8-config-nlp"

QWEN_ISVC_NAME = "qwen-isvc"

GORCH_NAME = "gorch-test"

USER_ONE: str = "user-one"
GUARDRAILS_ORCHESTRATOR_PORT: int = 8032


# GuardrailsOrchestrator related fixtures
@pytest.fixture(scope="class")
def guardrails_orchestrator_with_builtin_detectors(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    gorch_with_builtin_detectors_configmap: ConfigMap,
    guardrails_gateway_config: ConfigMap,
) -> Generator[GuardrailsOrchestrator, Any, Any]:
    with GuardrailsOrchestrator(
        client=admin_client,
        name=GORCH_NAME,
        namespace=model_namespace.name,
        enable_built_in_detectors=True,
        enable_guardrails_gateway=True,
        orchestrator_config=gorch_with_builtin_detectors_configmap.name,
        guardrails_gateway_config=guardrails_gateway_config.name,
        replicas=1,
        wait_for_resource=True,
    ) as gorch:
        orchestrator_deployment = Deployment(name=gorch.name, namespace=gorch.namespace, wait_for_resource=True)
        orchestrator_deployment.wait_for_replicas()
        yield gorch


@pytest.fixture(scope="class")
def gorch_with_builtin_detectors_pod(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    guardrails_orchestrator_with_builtin_detectors: GuardrailsOrchestrator,
) -> Pod:
    return list(Pod.get(namespace=model_namespace.name, label_selector=f"app.kubernetes.io/instance={GORCH_NAME}"))[0]


@pytest.fixture(scope="class")
def gorch_with_builtin_detectors_health_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    guardrails_orchestrator_with_builtin_detectors: GuardrailsOrchestrator,
) -> Generator[Route, Any, Any]:
    yield Route(
        name=f"{guardrails_orchestrator_with_builtin_detectors.name}-health",
        namespace=guardrails_orchestrator_with_builtin_detectors.namespace,
        wait_for_resource=True,
    )


@pytest.fixture(scope="class")
def gorch_with_builtin_detectors_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    guardrails_orchestrator_with_builtin_detectors: GuardrailsOrchestrator,
) -> Generator[Route, Any, Any]:
    yield Route(
        name=f"{guardrails_orchestrator_with_builtin_detectors.name}",
        namespace=guardrails_orchestrator_with_builtin_detectors.namespace,
        wait_for_resource=True,
    )


@pytest.fixture(scope="class")
def gorch_with_builtin_detectors_configmap(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[ConfigMap, Any, Any]:
    with ConfigMap(
        client=admin_client,
        name=ORCHESTRATOR_CONFIGMAP_NAME,
        namespace=model_namespace.name,
        data={
            "config.yaml": yaml.dump({
                "chat_generation": {
                    "service": {
                        "hostname": f"{QWEN_ISVC_NAME}-predictor.{model_namespace.name}.svc.cluster.local",
                        "port": GUARDRAILS_ORCHESTRATOR_PORT,
                    }
                },
                "detectors": {
                    "regex": {
                        "type": "text_contents",
                        "service": {
                            "hostname": "127.0.0.1",
                            "port": Ports.REST_PORT,
                        },
                        "chunker_id": "whole_doc_chunker",
                        "default_threshold": 0.5,
                    }
                },
            })
        },
    ) as cm:
        yield cm


@pytest.fixture(scope="class")
def guardrails_orchestrator_with_hf_detectors(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    gorch_with_hf_detectors_configmap: ConfigMap,
) -> Generator[GuardrailsOrchestrator, Any, Any]:
    with GuardrailsOrchestrator(
        client=admin_client,
        name=GORCH_NAME,
        namespace=model_namespace.name,
        enable_built_in_detectors=False,
        enable_guardrails_gateway=False,
        orchestrator_config=gorch_with_hf_detectors_configmap.name,
        replicas=1,
        wait_for_resource=True,
    ) as gorch:
        orchestrator_deployment = Deployment(name=gorch.name, namespace=gorch.namespace, wait_for_resource=True)
        orchestrator_deployment.wait_for_replicas()
        yield gorch


@pytest.fixture(scope="class")
def guardrails_orchestrator_with_hf_detectors_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    guardrails_orchestrator_with_hf_detectors: GuardrailsOrchestrator,
) -> Generator[Route, Any, Any]:
    yield Route(
        name=f"{guardrails_orchestrator_with_hf_detectors.name}",
        namespace=guardrails_orchestrator_with_hf_detectors.namespace,
        wait_for_resource=True,
    )


@pytest.fixture(scope="class")
def gorch_with_hf_detectors_configmap(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[ConfigMap, Any, Any]:
    with ConfigMap(
        client=admin_client,
        name=ORCHESTRATOR_CONFIGMAP_NAME,
        namespace=model_namespace.name,
        data={
            "config.yaml": yaml.dump({
                "chat_generation": {
                    "service": {
                        "hostname": f"{QWEN_ISVC_NAME}-predictor",
                        "port": GUARDRAILS_ORCHESTRATOR_PORT,
                    }
                },
                "detectors": {
                    "prompt_injection": {
                        "type": "text_contents",
                        "service": {
                            "hostname": "prompt-injection-detector-predictor",
                            "port": 8000,
                        },
                        "chunker_id": "whole_doc_chunker",
                        "default_threshold": 0.5,
                    }
                },
            })
        },
    ) as cm:
        yield cm


@pytest.fixture(scope="class")
def guardrails_gateway_config(
    admin_client: DynamicClient, model_namespace: Namespace
) -> Generator[ConfigMap, Any, Any]:
    with ConfigMap(
        client=admin_client,
        name="fms-orchestr8-config-gateway",
        namespace=model_namespace.name,
        label={Labels.Openshift.APP: "fmstack-nlp"},
        data={
            "config.yaml": yaml.dump({
                "orchestrator": {
                    "host": "localhost",
                    "port": GUARDRAILS_ORCHESTRATOR_PORT,
                },
                "detectors": [
                    {
                        "name": "regex",
                        "input": True,
                        "output": True,
                        "detector_params": {"regex": ["email", "ssn"]},
                    },
                    {
                        "name": "other_detector",
                        "input": True,
                        "output": True,
                    },
                ],
                "routes": [
                    {"name": "pii", "detectors": ["regex"]},
                    {"name": "passthrough", "detectors": []},
                ],
            })
        },
    ) as cm:
        yield cm


# ServingRuntimes, InferenceServices, and related resources
# for generation and detection models
@pytest.fixture(scope="class")
def vllm_runtime(
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
        "@sha256:d680ff8becb6bbaf83dfee7b2d9b8a2beb130db7fd5aa7f9a6d8286a58cebbfd",
        containers={
            "kserve-container": {
                "args": [
                    f"--port={str(GUARDRAILS_ORCHESTRATOR_PORT)}",
                    "--model=/mnt/models",
                ],
                "ports": [{"containerPort": GUARDRAILS_ORCHESTRATOR_PORT, "protocol": "TCP"}],
                "volumeMounts": [{"mountPath": "/dev/shm", "name": "shm"}],
            }
        },
        volumes=[{"emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"}, "name": "shm"}],
    ) as serving_runtime:
        yield serving_runtime


@pytest.fixture(scope="class")
def huggingface_sr(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    with ServingRuntime(
        client=admin_client,
        name="guardrails-detector-runtime-prompt-injection",
        namespace=model_namespace.name,
        containers=[
            {
                "name": "kserve-container",
                "image": "quay.io/trustyai/guardrails-detector-huggingface-runtime:v0.2.0",
                "command": ["uvicorn", "app:app"],
                "args": [
                    "--workers=4",
                    "--host=0.0.0.0",
                    "--port=8000",
                    "--log-config=/common/log_conf.yaml",
                ],
                "env": [
                    {"name": "MODEL_DIR", "value": "/mnt/models"},
                    {"name": "HF_HOME", "value": "/tmp/hf_home"},
                ],
                "ports": [{"containerPort": 8000, "protocol": "TCP"}],
            }
        ],
        supported_model_formats=[{"name": "guardrails-detector-huggingface", "autoSelect": True}],
        multi_model=False,
        annotations={
            "openshift.io/display-name": "Guardrails Detector ServingRuntime for KServe",
            "opendatahub.io/recommended-accelerators": '["nvidia.com/gpu"]',
            "prometheus.io/port": "8080",
            "prometheus.io/path": "/metrics",
        },
        label={
            "opendatahub.io/dashboard": "true",
        },
    ) as serving_runtime:
        yield serving_runtime


@pytest.fixture(scope="class")
def qwen_isvc(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    minio_pod: Pod,
    minio_service: Service,
    minio_data_connection: Secret,
    vllm_runtime: ServingRuntime,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name=QWEN_ISVC_NAME,
        namespace=model_namespace.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        model_format="vLLM",
        runtime=vllm_runtime.name,
        storage_key=minio_data_connection.name,
        storage_path="Qwen2.5-0.5B-Instruct",
        wait_for_predictor_pods=False,
        resources={
            "requests": {"cpu": "1", "memory": "8Gi"},
            "limits": {"cpu": "2", "memory": "10Gi"},
        },
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def prompt_injection_detector_isvc(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    minio_data_connection: Secret,
    huggingface_sr: ServingRuntime,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name="prompt-injection-detector",
        namespace=model_namespace.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        model_format="guardrails-detector-huggingface",
        runtime=huggingface_sr.name,
        storage_key=minio_data_connection.name,
        storage_path="deberta-v3-base-prompt-injection-v2",
        wait_for_predictor_pods=False,
        enable_auth=False,
        resources={
            "requests": {"cpu": "1", "memory": "2Gi", "nvidia.com/gpu": "0"},
            "limits": {"cpu": "1", "memory": "2Gi", "nvidia.com/gpu": "0"},
        },
        max_replicas=1,
        min_replicas=1,
        labels={
            "opendatahub.io/dashboard": "true",
        },
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def prompt_injection_detector_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    prompt_injection_detector_isvc: InferenceService,
) -> Generator[Route, Any, Any]:
    yield Route(
        name="prompt-injection-detector-route",
        namespace=model_namespace.name,
        service=prompt_injection_detector_isvc.name,
        wait_for_resource=True,
    )


# Llama-stack fixtures
@pytest.fixture(scope="class")
def lls_dist_gorch_with_builtin_detectors(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    qwen_isvc: InferenceService,
    guardrails_orchestrator_with_builtin_detectors: GuardrailsOrchestrator,
) -> Generator[LlamaStackDistribution, None, None]:
    with LlamaStackDistribution(
        name="llama-stack-distribution",
        namespace=model_namespace.name,
        replicas=1,
        server={
            "containerSpec": {
                "env": [
                    {
                        "name": "VLLM_URL",
                        "value": f"http://{qwen_isvc.name}-predictor.{model_namespace.name}.svc.cluster.local:8032/v1",
                    },
                    {
                        "name": "INFERENCE_MODEL",
                        "value": MNT_MODELS,
                    },
                    {
                        "name": "MILVUS_DB_PATH",
                        "value": "~/.llama/milvus.db",
                    },
                    {
                        "name": "VLLM_TLS_VERIFY",
                        "value": "false",
                    },
                    {
                        "name": "FMS_ORCHESTRATOR_URL",
                        "value": f"http://{guardrails_orchestrator_with_builtin_detectors.name}"
                        f"-service.{model_namespace.name}.svc.cluster.local:8090",
                    },
                ],
                "name": "llama-stack",
                "port": 8321,
            },
            "distribution": {"image": "quay.io/opendatahub/llama-stack:odh"},
            "storage": {
                "size": "20Gi",
            },
        },
        wait_for_resource=True,
    ) as lls_dist:
        lls_dist.wait_for_status(status=LlamaStackDistribution.Status.READY)
        yield lls_dist


@pytest.fixture(scope="class")
def lls_dist_gorch_with_builtin_detectors_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    lls_dist_gorch_with_builtin_detectors: LlamaStackDistribution,
) -> Generator[Service, None, None]:
    yield Service(
        client=admin_client,
        name=f"{lls_dist_gorch_with_builtin_detectors.name}-service",
        namespace=model_namespace.name,
        wait_for_resource=True,
    )


@pytest.fixture(scope="class")
def lls_dist_gorch_with_builtin_detectors_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    lls_dist_gorch_with_builtin_detectors: LlamaStackDistribution,
    lls_dist_gorch_with_builtin_detectors_service: Service,
) -> Generator[Route, None, None]:
    with Route(
        client=admin_client,
        name=f"{lls_dist_gorch_with_builtin_detectors.name}-route",
        namespace=model_namespace.name,
        service=lls_dist_gorch_with_builtin_detectors_service.name,
    ) as route:
        yield route


@pytest.fixture(scope="class")
def lls_client(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    lls_dist_gorch_with_builtin_detectors: LlamaStackDistribution,
    lls_dist_gorch_with_builtin_detectors_route: Route,
) -> LlamaStackClient:
    return LlamaStackClient(base_url=f"http://{lls_dist_gorch_with_builtin_detectors_route.host}")


# Other
@pytest.fixture(scope="class")
def openshift_ca_bundle_file(
    admin_client: DynamicClient,
) -> str:
    return create_ca_bundle_file(client=admin_client, ca_type="openshift")
