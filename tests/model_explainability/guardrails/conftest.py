from typing import Generator, Any

import pytest
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.guardrails_orchestrator import GuardrailsOrchestrator
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.role_binding import RoleBinding
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime

from utilities.constants import Labels, Annotations, KServeDeploymentType, Timeout
from utilities.inference_utils import create_isvc

MINIO_LLM: str = "minio-llm"
USER_ONE: str = "user-one"


@pytest.fixture(scope="class")
def guardrails_orchestrator_health_route(
    admin_client: DynamicClient, model_namespace: Namespace, guardrails_orchestrator: GuardrailsOrchestrator
) -> Generator[Route, Any, Any]:
    route = Route(
        name=f"{guardrails_orchestrator.name}-health",
        namespace=guardrails_orchestrator.namespace,
        wait_for_resource=True,
    )
    yield route


@pytest.fixture(scope="class")
def guardrails_orchestrator(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    orchestrator_configmap: ConfigMap,
    vllm_gateway_config: ConfigMap,
    vllm_images_configmap: ConfigMap,
) -> Generator[GuardrailsOrchestrator, Any, Any]:
    with GuardrailsOrchestrator(
        client=admin_client,
        name="gorch-test",
        namespace=model_namespace.name,
        orchestrator_config=orchestrator_configmap.name,
        vllm_gateway_config=vllm_gateway_config.name,
        replicas=1,
        wait_for_resource=True,
    ) as gorch:
        orchestrator_deployment = Deployment(name=gorch.name, namespace=gorch.namespace, wait_for_resource=True)
        orchestrator_deployment.wait_for_replicas()
        yield gorch


@pytest.fixture(scope="class")
def qwen_llm_model(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    minio_llm_data_connection: Secret,
    vllm_runtime: ServingRuntime,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name="llm",
        namespace=model_namespace.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        model_format="vLLM",
        runtime=vllm_runtime.name,
        storage_key=minio_llm_data_connection.name,
        storage_path="Qwen2.5-0.5B-Instruct",
        wait_for_predictor_pods=False,
        enable_auth=True,
        resources={"requests": {"cpu": "1", "memory": "8Gi"}, "limits": {"cpu": "2", "memory": "10Gi"}},
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def vllm_runtime(
    admin_client: DynamicClient, model_namespace: Namespace, minio_llm_data_connection: Secret
) -> Generator[ServingRuntime, Any, Any]:
    with ServingRuntime(
        client=admin_client,
        name="vllm-runtime-cpu-fp16",
        namespace=model_namespace.name,
        containers=[
            {
                "name": "kserve-container",
                "image": "quay.io/rh-aiservices-bu/vllm-cpu-openai-ubi9"
                "@sha256:d680ff8becb6bbaf83dfee7b2d9b8a2beb130db7fd5aa7f9a6d8286a58cebbfd",
                "command": ["python", "-m", "vllm.entrypoints.openai.api_server"],
                "args": [
                    "--port=8032",
                    "--model=/mnt/models",
                    "--served-model-name={{.Name}}",
                    "--dtype=float16",
                    "--enforce-eager",
                ],
                "env": [
                    {"name": "HF_HOME", "value": "/tmp/hf_home"},
                ],
                "ports": [
                    {"containerPort": 8032, "protocol": "TCP"},
                ],
                "volumeMounts": [
                    {"name": "shm", "mountPath": "/dev/shm"},
                ],
            }
        ],
        supported_model_formats=[
            {"name": "vLLM", "autoSelect": True},
        ],
        multi_model=False,
        volumes=[
            {
                "name": "shm",
                "emptyDir": {
                    "medium": "Memory",
                    "sizeLimit": "2Gi",
                },
            }
        ],
        annotations={
            "openshift.io/display-name": "vLLM ServingRuntime for KServe - Float16",
        },
        spec_annotations={
            "prometheus.io/path": "/metrics",
            "prometheus.io/port": "8080",
        },
        label={Labels.OpenDataHub.DASHBOARD: "true"},
    ) as vllm_runtime:
        yield vllm_runtime


@pytest.fixture(scope="class")
def vllm_images_configmap(admin_client: DynamicClient, model_namespace: Namespace) -> Generator[ConfigMap, Any, Any]:
    with ConfigMap(
        client=admin_client,
        name="gorch-test-config",
        namespace=model_namespace.name,
        data={
            "regexDetectorImage": "quay.io/trustyai_testing/regex-detector"
            "@sha256:e9df9f7e7429e29da9b8d9920d80cdc85a496e7961f6edb19132d604a914049b",
            "vllmGatewayImage": "quay.io/trustyai_testing/vllm-orchestrator-gateway"
            "@sha256:d0bbf2de95c69f76215a016820f294202c48721dee452b3939e36133697d5b1d",
        },
    ) as cm:
        yield cm


@pytest.fixture(scope="class")
def orchestrator_configmap(
    admin_client: DynamicClient, model_namespace: Namespace, qwen_llm_model: InferenceService
) -> Generator[ConfigMap, Any, Any]:
    with ConfigMap(
        client=admin_client,
        name="fms-orchestr8-config-nlp",
        namespace=model_namespace.name,
        data={
            "config.yaml": yaml.dump({
                "chat_generation": {
                    "service": {
                        "hostname": f"{qwen_llm_model.name}-predictor.{model_namespace.name}.svc.cluster.local",
                        "port": 8032,
                    }
                },
                "detectors": {
                    "regex": {
                        "type": "text_contents",
                        "service": {"hostname": "127.0.0.1", "port": 8080},
                        "chunker_id": "whole_doc_chunker",
                        "default_threshold": 0.5,
                    }
                },
            })
        },
    ) as cm:
        yield cm


@pytest.fixture(scope="class")
def vllm_gateway_config(admin_client: DynamicClient, model_namespace: Namespace) -> Generator[ConfigMap, Any, Any]:
    with ConfigMap(
        client=admin_client,
        name="fms-orchestr8-config-gateway",
        namespace=model_namespace.name,
        label={"app": "fmstack-nlp"},
        data={
            "config.yaml": yaml.dump({
                "orchestrator": {"host": "localhost", "port": 8032},
                "detectors": [
                    {"name": "regex", "detector_params": {"regex": ["email", "ssn"]}},
                    {"name": "other_detector"},
                ],
                "routes": [{"name": "pii", "detectors": ["regex"]}, {"name": "passthrough", "detectors": []}],
            })
        },
    ) as cm:
        yield cm


@pytest.fixture(scope="class")
def minio_llm_deployment(
    admin_client: DynamicClient, model_namespace: Namespace, llm_models_pvc: PersistentVolumeClaim
) -> Generator[Deployment, Any, Any]:
    with Deployment(
        client=admin_client,
        name="llm-container-deployment",
        namespace=model_namespace.name,
        replicas=1,
        selector={"matchLabels": {"app": MINIO_LLM}},
        template={
            "metadata": {"labels": {"app": MINIO_LLM, "maistra.io/expose-route": "true"}, "name": MINIO_LLM},
            "spec": {
                "volumes": [{"name": "model-volume", "persistentVolumeClaim": {"claimName": "llm-models-claim"}}],
                "initContainers": [
                    {
                        "name": "download-model",
                        "image": "quay.io/trustyai_testing/llm-downloader-bootstrap"
                        "@sha256:d3211cc581fe69ca9a1cb75f84e5d08cacd1854cb2d63591439910323b0cbb57",
                        "securityContext": {"fsGroup": 1001},
                        "command": [
                            "bash",
                            "-c",
                            'model="Qwen/Qwen2.5-0.5B-Instruct"'
                            '\necho "starting download"'
                            "\n/tmp/venv/bin/huggingface-cli download $model "
                            "--local-dir /mnt/models/llms/$(basename $model)"
                            '\necho "Done!"',
                        ],
                        "resources": {"limits": {"memory": "5Gi", "cpu": "2"}},
                        "volumeMounts": [{"mountPath": "/mnt/models/", "name": "model-volume"}],
                    }
                ],
                "containers": [
                    {
                        "args": ["server", "/models"],
                        "env": [
                            {"name": "MINIO_ACCESS_KEY", "value": "THEACCESSKEY"},
                            {"name": "MINIO_SECRET_KEY", "value": "THESECRETKEY"},
                        ],
                        "image": "quay.io/trustyai/modelmesh-minio-examples"
                        "@sha256:65cb22335574b89af15d7409f62feffcc52cc0e870e9419d63586f37706321a5",
                        "name": MINIO_LLM,
                        "securityContext": {
                            "allowPrivilegeEscalation": False,
                            "capabilities": {"drop": ["ALL"]},
                            "seccompProfile": {"type": "RuntimeDefault"},
                        },
                        "volumeMounts": [{"mountPath": "/models/", "name": "model-volume"}],
                    }
                ],
            },
        },
        label={"app": MINIO_LLM},
        wait_for_resource=True,
    ) as deployment:
        deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_10MIN)
        yield deployment


@pytest.fixture(scope="class")
def llm_models_pvc(
    admin_client: DynamicClient, model_namespace: Namespace
) -> Generator[PersistentVolumeClaim, Any, Any]:
    with PersistentVolumeClaim(
        client=admin_client,
        name="llm-models-claim",
        namespace=model_namespace.name,
        accessmodes=PersistentVolumeClaim.AccessMode.RWO,
        volume_mode=PersistentVolumeClaim.VolumeMode.FILE,
        size="10Gi",
    ) as pvc:
        yield pvc


@pytest.fixture(scope="class")
def minio_llm_service(admin_client: DynamicClient, model_namespace: Namespace) -> Generator[Service, Any, Any]:
    with Service(
        client=admin_client,
        name=MINIO_LLM,
        namespace=model_namespace.name,
        ports=[{"name": "minio-client-port", "port": 9000, "protocol": "TCP", "targetPort": 9000}],
        selector={"app": MINIO_LLM},
    ) as service:
        yield service


@pytest.fixture(scope="class")
def minio_llm_data_connection(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    minio_llm_deployment: Deployment,
    minio_llm_service: Service,
) -> Generator[Secret, Any, Any]:
    with Secret(
        client=admin_client,
        name="aws-connection-minio-data-connection",
        namespace=model_namespace.name,
        data_dict={
            # Dummy AWS values
            "AWS_ACCESS_KEY_ID": "VEhFQUNDRVNTS0VZ",
            "AWS_DEFAULT_REGION": "dXMtc291dGg=",
            "AWS_S3_BUCKET": "bGxtcw==",
            "AWS_S3_ENDPOINT": "aHR0cDovL21pbmlvLWxsbTo5MDAw",
            "AWS_SECRET_ACCESS_KEY": "VEhFU0VDUkVUS0VZ",  # pragma: allowlist secret
        },
        label={Labels.OpenDataHub.DASHBOARD: "true", Annotations.OpenDataHubIo.MANAGED: "true"},
        annotations={"opendatahub.io/connection-type": "s3", "openshift.io/display-name": "Minio LLM Data Connection"},
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def user_one_service_account(
    admin_client: DynamicClient, model_namespace: Namespace
) -> Generator[ServiceAccount, Any, Any]:
    with ServiceAccount(client=admin_client, name=USER_ONE, namespace=model_namespace.name) as service_account:
        yield service_account


@pytest.fixture(scope="class")
def user_one_rolebinding(
    admin_client: DynamicClient, model_namespace: Namespace, user_one_service_account: ServiceAccount
) -> Generator[RoleBinding, Any, Any]:
    with RoleBinding(
        client=admin_client,
        name=f"{user_one_service_account.name}-view",
        namespace=model_namespace.name,
        subjects=[{"kind": "ServiceAccount", "name": user_one_service_account.name}],
        role_ref={"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "view"},
    ) as role_binding:
        yield role_binding
