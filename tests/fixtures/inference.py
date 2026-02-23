from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.serving_runtime import ServingRuntime
from pytest_testconfig import py_config
from simple_logger.logger import get_logger
from timeout_sampler import retry

from utilities.constants import (
    QWEN_MODEL_NAME,
    KServeDeploymentType,
    LLMdInferenceSimConfig,
    RuntimeTemplates,
)
from utilities.inference_utils import create_isvc
from utilities.infra import get_data_science_cluster, wait_for_dsc_status_ready
from utilities.serving_runtime import ServingRuntimeFromTemplate

LOGGER = get_logger(name=__name__)


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
        enable_auth=False,
        resources={
            "requests": {"cpu": "1", "memory": "6Gi"},
            "limits": {"cpu": "2", "memory": "12Gi"},
        },
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def qwen_isvc_url(qwen_isvc: InferenceService) -> str:
    return f"http://{qwen_isvc.name}-predictor.{qwen_isvc.namespace}.svc.cluster.local:8032/v1"


@pytest.fixture(scope="class")
def llm_d_inference_sim_serving_runtime(
    admin_client: DynamicClient, model_namespace: Namespace, teardown_resources: bool, pytestconfig: pytest.Config
) -> Generator[ServingRuntime, Any, Any]:
    """Serving runtime for LLM-d Inference Simulator.

    While llm-d-inference-sim supports any model name, the /tokenizers endpoint will only support two models
        - qwen2.5-0.5b-instruct
        - Qwen2.5-1.5B-Instruct

    For other models, ensure:
        - the correct write permissions on the Pod
        - the model name matches what is available on HuggingFace (e.g., Qwen/Qwen2.5-1.5B-Instruct)
        - you have set a writeable "--tokenizers-cache-dir"
        - the cluster can pull from HuggingFace

    """
    if pytestconfig.option.post_upgrade:
        sr = ServingRuntime(
            client=admin_client, name=LLMdInferenceSimConfig.serving_runtime_name, namespace=model_namespace.name
        )
        yield sr
        sr.clean_up()

    else:
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
                    "image": "quay.io/trustyai_testing/llm-d-inference-sim-dataset-builtin"
                    "@sha256:79e525cfd57a0d72b7e71d5f1e2dd398eca9315cfbd061d9d3e535b1ae736239",
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
            teardown=teardown_resources,
        ) as serving_runtime:
            yield serving_runtime


@pytest.fixture(scope="class")
def llm_d_inference_sim_isvc(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    llm_d_inference_sim_serving_runtime: ServingRuntime,
    teardown_resources: bool,
    pytestconfig: pytest.Config,
) -> Generator[InferenceService, Any, Any]:
    """Fixture for LLMdInferenceSim InferenceService."""
    if pytestconfig.option.post_upgrade:
        isvc = InferenceService(
            client=admin_client, name=LLMdInferenceSimConfig.isvc_name, namespace=model_namespace.name
        )
        yield isvc
        isvc.clean_up()
    else:
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
            teardown=teardown_resources,
        ) as isvc:
            yield isvc


@pytest.fixture(scope="class")
def kserve_controller_manager_deployment(admin_client: DynamicClient) -> Generator[Deployment, Any, Any]:
    yield Deployment(
        client=admin_client,
        name="kserve-controller-manager",
        namespace=py_config["applications_namespace"],
        ensure_exists=True,
    )


@pytest.fixture(scope="class")
def patched_dsc_kserve_headed(
    admin_client, kserve_controller_manager_deployment: Deployment
) -> Generator[DataScienceCluster]:
    """Configure KServe Services to work in Headed mode i.e. using the Service port instead of the Pod port"""

    def _kserve_status(dsc_resource: DataScienceCluster) -> str:
        return next(
            filter(lambda condition: condition["type"] == "KserveReady", dsc_resource.instance.status["conditions"])
        )["status"]

    @retry(wait_timeout=30, sleep=1)
    def _wait_for_kserve_upgrade(dsc_resource: DataScienceCluster):
        return not _kserve_status(dsc_resource) == "True"

    dsc = get_data_science_cluster(client=admin_client)
    if not dsc.instance.spec.components.kserve.rawDeploymentServiceConfig == "Headed":
        with ResourceEditor(
            patches={dsc: {"spec": {"components": {"kserve": {"rawDeploymentServiceConfig": "Headed"}}}}}
        ):
            _wait_for_kserve_upgrade(dsc_resource=dsc)
            kserve_controller_manager_deployment.wait_for_replicas()
            wait_for_dsc_status_ready(dsc_resource=dsc)
            yield dsc
    else:
        LOGGER.info("DSC already configured for Headed mode")
        yield dsc
