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
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.serving_runtime import ServingRuntime
from pytest_testconfig import config as py_config

from tests.model_explainability.guardrails.conftest import GUARDRAILS_ORCHESTRATOR_NAME
from tests.model_explainability.guardrails.constants import QWEN_ISVC_NAME
from tests.model_explainability.guardrails.test_guardrails import MNT_MODELS
from tests.model_explainability.trustyai_service.trustyai_service_utils import TRUSTYAI_SERVICE_NAME
from utilities.constants import RuntimeTemplates, KServeDeploymentType, Labels
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


# Guardrails Orchestrator Fixtures
@pytest.fixture(scope="class")
def guardrails_orchestrator(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    orchestrator_config: ConfigMap,
) -> Generator[GuardrailsOrchestrator, Any, Any]:
    gorch_kwargs = {
        "client": admin_client,
        "name": GUARDRAILS_ORCHESTRATOR_NAME,
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
    return list(
        Pod.get(
            namespace=model_namespace.name, label_selector=f"app.kubernetes.io/instance={GUARDRAILS_ORCHESTRATOR_NAME}"
        )
    )[0]


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


# LlamaStack fixtures
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


@pytest.fixture(scope="class")
def guardrails_orchestrator_ssl_cert(guardrails_orchestrator_route: Route):
    hostname = guardrails_orchestrator_route.host

    try:
        result = subprocess.run(
            args=["openssl", "s_client", "-showcerts", "-connect", f"{hostname}:443"],
            input="",
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0 and "CONNECTED" not in result.stdout:
            raise RuntimeError(f"Failed to connect to {hostname}: {result.stderr}")

        cert_lines = []
        in_cert = False
        for line in result.stdout.splitlines():
            if "-----BEGIN CERTIFICATE-----" in line:
                in_cert = True
            if in_cert:
                cert_lines.append(line)
            if "-----END CERTIFICATE-----" in line:
                in_cert = False

        if not cert_lines:
            raise RuntimeError(f"No certificate found in response from {hostname}")

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
