import os
import subprocess
from base64 import b64encode
from typing import Generator, Any

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from llama_stack_client import LlamaStackClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.guardrails_orchestrator import GuardrailsOrchestrator
from ocp_resources.inference_service import InferenceService
from ocp_resources.llama_stack_distribution import LlamaStackDistribution
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.serving_runtime import ServingRuntime
from pytest_testconfig import py_config

from tests.model_explainability.guardrails.constants import QWEN_ISVC_NAME
from tests.model_explainability.guardrails.test_guardrails import MNT_MODELS
from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import (
    KServeDeploymentType,
    Labels,
    RuntimeTemplates,
)
from utilities.inference_utils import create_isvc
from utilities.serving_runtime import ServingRuntimeFromTemplate


USER_ONE: str = "user-one"

GORCH_NAME = "guardrails-orchestrator"
GORCH_CONFIGMAP_NAME = "fms-orchestr8-config-nlp"
GORCH_PORT: int = 8032


# GuardrailsOrchestrator related fixtures
@pytest.fixture(scope="class")
def guardrails_orchestrator(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    orchestrator_config: ConfigMap,
) -> Generator[GuardrailsOrchestrator, Any, Any]:
    gorch_kwargs = {
        "client": admin_client,
        "name": GORCH_NAME,
        "namespace": model_namespace.name,
        "orchestrator_config": orchestrator_config.name,
        "replicas": 1,
        "wait_for_resource": True,
    }

    if enable_built_in_detectors := request.param.get("enable_built_in_detectors"):
        gorch_kwargs["enable_built_in_detectors"] = enable_built_in_detectors

    if request.param.get("enable_guardrails_gateway"):
        guardrails_gateway_config = request.getfixturevalue(argname="guardrails_gateway_config")
        gorch_kwargs["enable_guardrails_gateway"] = True
        gorch_kwargs["guardrails_gateway_config"] = guardrails_gateway_config.name

    with GuardrailsOrchestrator(**gorch_kwargs) as gorch:
        gorch_deployment = Deployment(name=gorch.name, namespace=gorch.namespace, wait_for_resource=True)
        gorch_deployment.wait_for_replicas()
        yield gorch


@pytest.fixture(scope="class")
def orchestrator_config(
    request: FixtureRequest, admin_client: DynamicClient, model_namespace: Namespace
) -> Generator[ConfigMap, Any, Any]:
    with ConfigMap(
        client=admin_client,
        name="fms-orchestr8-config-nlp",
        namespace=model_namespace.name,
        data=request.param["orchestrator_config_data"],
    ) as cm:
        yield cm


@pytest.fixture(scope="class")
def guardrails_gateway_config(
    request: FixtureRequest, admin_client: DynamicClient, model_namespace: Namespace
) -> Generator[ConfigMap, Any, Any]:
    with ConfigMap(
        client=admin_client,
        name="fms-orchestr8-config-gateway",
        namespace=model_namespace.name,
        label={Labels.Openshift.APP: "fmstack-nlp"},
        data=request.param["guardrails_gateway_config_data"],
    ) as cm:
        yield cm


@pytest.fixture(scope="class")
def guardrails_orchestrator_pod(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    guardrails_orchestrator: GuardrailsOrchestrator,
) -> Pod:
    return list(Pod.get(namespace=model_namespace.name, label_selector=f"app.kubernetes.io/instance={GORCH_NAME}"))[0]


@pytest.fixture(scope="class")
def guardrails_orchestrator_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    guardrails_orchestrator: GuardrailsOrchestrator,
) -> Generator[Route, Any, Any]:
    yield Route(
        name=f"{guardrails_orchestrator.name}",
        namespace=guardrails_orchestrator.namespace,
        wait_for_resource=True,
    )


@pytest.fixture(scope="class")
def guardrails_orchestrator_health_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    guardrails_orchestrator: GuardrailsOrchestrator,
) -> Generator[Route, Any, Any]:
    yield Route(
        name=f"{guardrails_orchestrator.name}-health",
        namespace=guardrails_orchestrator.namespace,
        wait_for_resource=True,
    )


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
                    f"--port={str(GORCH_PORT)}",
                    "--model=/mnt/models",
                ],
                "ports": [{"containerPort": GORCH_PORT, "protocol": "TCP"}],
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
def llamastack_distribution_trustyai(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    qwen_isvc: InferenceService,
    guardrails_orchestrator_route: Route,
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
                        "value": f"https://{guardrails_orchestrator_route.host}",
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
        lls_dist.wait_for_status(status=LlamaStackDistribution.Status.READY)
        yield lls_dist


@pytest.fixture(scope="class")
def llamastack_distribution_trustyai_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    llamastack_distribution_trustyai: LlamaStackDistribution,
) -> Generator[Service, None, None]:
    yield Service(
        client=admin_client,
        name=f"{llamastack_distribution_trustyai.name}-service",
        namespace=model_namespace.name,
        wait_for_resource=True,
    )


@pytest.fixture(scope="class")
def llamastack_distribution_trustyai_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    llamastack_distribution_trustyai: LlamaStackDistribution,
    llamastack_distribution_trustyai_service: Service,
) -> Generator[Route, None, None]:
    with Route(
        client=admin_client,
        name=f"{llamastack_distribution_trustyai.name}-route",
        namespace=model_namespace.name,
        service=llamastack_distribution_trustyai_service.name,
    ) as route:
        yield route


@pytest.fixture(scope="class")
def llamastack_client_trustyai(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    llamastack_distribution_trustyai_route: Route,
) -> LlamaStackClient:
    return LlamaStackClient(base_url=f"http://{llamastack_distribution_trustyai_route.host}")


# Other
@pytest.fixture(scope="class")
def openshift_ca_bundle_file(
    admin_client: DynamicClient,
) -> str:
    return create_ca_bundle_file(client=admin_client, ca_type="openshift")


@pytest.fixture(scope="class")
def guardrails_orchestrator_ssl_cert(guardrails_orchestrator_route: Route):
    hostname = guardrails_orchestrator_route.host

    try:
        cmd = ["openssl", "s_client", "-showcerts", "-connect", f"{hostname}:443"]

        process = subprocess.Popen(
            args=cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        stdout, _ = process.communicate(input="")

        cert_lines = []
        in_cert = False
        for line in stdout.split("\n"):
            if "-----BEGIN CERTIFICATE-----" in line:
                in_cert = True
            if in_cert:
                cert_lines.append(line)
            if "-----END CERTIFICATE-----" in line:
                in_cert = False

        filepath = os.path.join(py_config["tmp_base_dir"], "gorch_cert.crt")
        with open(filepath, "w") as f:
            f.write("\n".join(cert_lines))

        return filepath

    except Exception as e:
        raise RuntimeError(f"Could not get certificate from {hostname}: {e}")


@pytest.fixture(scope="class")
def guardrails_orchestrator_ssl_cert_secret(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    guardrails_orchestrator_ssl_cert: str,  # ← Add dependency and use correct cert
) -> Generator[Secret, Any, None]:
    with open(guardrails_orchestrator_ssl_cert, "r") as f:
        cert_content = f.read()

    with Secret(
        client=admin_client,
        name="orch-certificate",
        namespace=model_namespace.name,
        data_dict={"orch-certificate.crt": b64encode(cert_content.encode("utf-8")).decode("utf-8")},
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def patched_llamastack_deployment_tls_certs(llamastack_distribution_trustyai, guardrails_orchestrator_ssl_cert_secret):
    lls_deployment = Deployment(
        name=llamastack_distribution_trustyai.name,
        namespace=llamastack_distribution_trustyai.namespace,
        ensure_exists=True,
    )

    current_spec = lls_deployment.instance.spec.template.spec.to_dict()

    current_spec["volumes"].append({
        "name": "router-ca",
        "secret": {"secretName": "orch-certificate"},  # pragma: allowlist secret
    })

    for container in current_spec["containers"]:
        if container["name"] == "llama-stack":
            container["volumeMounts"].append({"name": "router-ca", "mountPath": "/etc/llama/certs", "readOnly": True})
            break

    with ResourceEditor(patches={lls_deployment: {"spec": {"template": {"spec": current_spec}}}}) as _:
        initial_replicas = lls_deployment.replicas
        lls_deployment.scale_replicas(replica_count=0)
        lls_deployment.scale_replicas(replica_count=initial_replicas)
        lls_deployment.wait_for_replicas()
        yield lls_deployment
