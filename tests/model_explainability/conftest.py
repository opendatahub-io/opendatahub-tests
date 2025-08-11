from typing import Generator, Any

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from llama_stack_client import LlamaStackClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.inference_service import InferenceService
from ocp_resources.llama_stack_distribution import LlamaStackDistribution
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.serving_runtime import ServingRuntime
from pytest_testconfig import config as py_config

from tests.model_explainability.guardrails.constants import QWEN_ISVC_NAME
from tests.model_explainability.constants import MNT_MODELS
from tests.model_explainability.trustyai_service.trustyai_service_utils import TRUSTYAI_SERVICE_NAME
from utilities.constants import KServeDeploymentType, RuntimeTemplates
from utilities.inference_utils import create_isvc
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="class")
def pvc_minio_namespace(
    admin_client: DynamicClient, minio_namespace: Namespace
) -> Generator[PersistentVolumeClaim, Any, Any]:
    with PersistentVolumeClaim(
        client=admin_client,
        name="minio-pvc",
        namespace=minio_namespace.name,
        accessmodes=PersistentVolumeClaim.AccessMode.RWO,
        volume_mode=PersistentVolumeClaim.VolumeMode.FILE,
        size="10Gi",
    ) as pvc:
        yield pvc


@pytest.fixture(scope="session")
def trustyai_operator_configmap(
    admin_client: DynamicClient,
) -> ConfigMap:
    return ConfigMap(
        client=admin_client,
        namespace=py_config["applications_namespace"],
        name=f"{TRUSTYAI_SERVICE_NAME}-operator-config",
        ensure_exists=True,
    )


# LlamaStack fixtures
@pytest.fixture(scope="class")
def llamastack_distribution(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    qwen_isvc: InferenceService,
) -> Generator[LlamaStackDistribution, None, None]:
    fms_orchestrator_url = ""
    if hasattr(request, "param") and request.param.get("guardrails_orchestrator_route_fixture"):
        guardrails_orchestrator_route_fixture_name = request.param.get("guardrails_orchestrator_route_fixture")
        guardrails_orchestrator_route = request.getfixturevalue(argname=guardrails_orchestrator_route_fixture_name)
        fms_orchestrator_url = f"https://{guardrails_orchestrator_route.host}"

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
                        "value": fms_orchestrator_url,
                    },
                ],
                "name": "llama-stack",
                "port": 8321,
            },
            "distribution": {"name": "rh-dev"},
            "storage": {
                "size": "20Gi",
            },
        },
        wait_for_resource=True,
    ) as lls_dist:
        lls_dist.wait_for_status(status=LlamaStackDistribution.Status.READY, timeout=3600)
        yield lls_dist


@pytest.fixture(scope="class")
def llamastack_distribution_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    llamastack_distribution: LlamaStackDistribution,
) -> Generator[Service, None, None]:
    yield Service(
        client=admin_client,
        name=f"{llamastack_distribution.name}-service",
        namespace=model_namespace.name,
        wait_for_resource=True,
    )


@pytest.fixture(scope="class")
def llamastack_distribution_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    llamastack_distribution: LlamaStackDistribution,
    llamastack_distribution_service: Service,
) -> Generator[Route, None, None]:
    with Route(
        client=admin_client,
        name=f"{llamastack_distribution.name}-route",
        namespace=model_namespace.name,
        service=llamastack_distribution_service.name,
    ) as route:
        yield route


@pytest.fixture(scope="class")
def llamastack_client(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    llamastack_distribution_route: Route,
) -> LlamaStackClient:
    return LlamaStackClient(base_url=f"http://{llamastack_distribution_route.host}")


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
                    f"--port={str(8032)}",
                    "--model=/mnt/models",
                ],
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
