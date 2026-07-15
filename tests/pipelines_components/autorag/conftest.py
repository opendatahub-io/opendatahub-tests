import os
import shlex
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import httpx
import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.data_science_pipelines_application import DataSciencePipelinesApplication
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount
from ogx_client import APIConnectionError, InternalServerError, OgxClient
from timeout_sampler import TimeoutExpiredError, TimeoutSampler, retry

from tests.fixtures.vector_io import (  # noqa: NIT001
    MILVUS_TOKEN,
    get_etcd_deployment_template,
    get_milvus_deployment_template,
)
from tests.pipelines_components.constants import (
    AUTORAG_EMBEDDING_MAX_MODEL_LEN,
    AUTORAG_INPUT_DATA_KEY,
    AUTORAG_LLAMA_STACK_INFERENCE_MODEL_ID,
    AUTORAG_MAX_RAG_PATTERNS,
    AUTORAG_OPTIMIZATION_METRIC,
    AUTORAG_PIPELINE_YAML,
    AUTORAG_S3_BUCKET,
    AUTORAG_TEST_DATA_KEY,
    DSPA_NAME,
    DSPA_READY_BUFFER_SECONDS,
    DSPA_S3_BUCKET,
    DSPA_S3_SECRET,
    MANAGED_PIPELINE_AUTORAG,
    MANAGED_PIPELINE_POLL_INTERVAL,
    MANAGED_PIPELINE_WAIT_TIMEOUT,
    MINIO_MC_IMAGE,
    MINIO_UPLOADER_SECURITY_CONTEXT,
)
from tests.pipelines_components.utils import (
    create_pipeline_run,
    create_pipeline_run_managed,
    delete_pipeline,
    delete_pipeline_run,
    resolve_pipeline_yaml,
    upload_pipeline,
    use_managed_pipelines,
    wait_for_managed_pipeline,
)
from utilities.constants import Annotations, DscComponents, KServeDeploymentType, RuntimeTemplates
from utilities.data_science_cluster_utils import update_components_in_dsc
from utilities.exceptions import UnexpectedResourceCountError
from utilities.general import generate_random_name
from utilities.inference_utils import create_isvc
from utilities.infra import create_ns
from utilities.resources.ogx_server import OgxServer
from utilities.serving_runtime import ServingRuntimeFromTemplate

LOGGER = structlog.get_logger(name=__name__)

AUTORAG_RESOURCE_PREFIX: str = "autorag-smoke"

OGX_CLIENT_VERIFY_SSL: bool = os.getenv("OGX_CLIENT_VERIFY_SSL", "false").lower() == "true"
OGX_CORE_POD_FILTER: str = "app=ogx"
POSTGRES_IMAGE: str = os.getenv(
    "OGX_VECTOR_IO_POSTGRES_IMAGE",
    (
        "registry.redhat.io/rhel9/postgresql-15@sha256:"
        "90ec347a35ab8a5d530c8d09f5347b13cc71df04f3b994bfa8b1a409b1171d59"  # pragma: allowlist secret
    ),
)

# User-provided env vars for the models to deploy
AUTORAG_INFERENCE_MODEL_URI: str = os.environ.get("AUTORAG_INFERENCE_MODEL_URI", "")
AUTORAG_INFERENCE_MODEL_NAME: str = os.environ.get("AUTORAG_INFERENCE_MODEL_NAME", "")
AUTORAG_EMBEDDING_MODEL_URI: str = os.environ.get("AUTORAG_EMBEDDING_MODEL_URI", "")
AUTORAG_EMBEDDING_MODEL_NAME: str = os.environ.get("AUTORAG_EMBEDDING_MODEL_NAME", "")

_AUTORAG_REQUIRED_ENV = {
    "AUTORAG_INFERENCE_MODEL_URI": "Storage URI for inference model (e.g. s3://bucket/model or hf://org/model)",
    "AUTORAG_INFERENCE_MODEL_NAME": "Inference model name (e.g. granite-3b-instruct)",
    "AUTORAG_EMBEDDING_MODEL_URI": "Storage URI for embedding model (e.g. s3://bucket/model or hf://org/model)",
    "AUTORAG_EMBEDDING_MODEL_NAME": "Embedding model name (e.g. bge-m3)",
}

AUTORAG_OGX_SECRET_DATA: dict[str, str] = {
    "postgres-user": os.getenv("OGX_VECTOR_IO_POSTGRESQL_USER", "ps_user"),
    "postgres-password": os.getenv("OGX_VECTOR_IO_POSTGRESQL_PASSWORD", "ps_password"),
    "vllm-api-token": os.getenv("OGX_CORE_VLLM_API_TOKEN", ""),
    "vllm-embedding-api-token": os.getenv("OGX_CORE_VLLM_EMBEDDING_API_TOKEN", "fake"),
    "milvus-token": MILVUS_TOKEN,
    "aws-access-key-id": os.getenv("AWS_ACCESS_KEY_ID", ""),
    "aws-secret-access-key": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
}


@contextmanager
def _create_ogx_server(
    client: DynamicClient,
    name: str,
    namespace: str,
    config: dict[str, Any],
) -> Generator[OgxServer, Any, Any]:
    with OgxServer(
        client=client,
        name=name,
        namespace=namespace,
        distribution=config["distribution"],
        workload=config.get("workload"),
        tls=config.get("tls"),
        wait_for_resource=True,
    ) as ogx_srv:
        yield ogx_srv


@retry(
    wait_timeout=240,
    sleep=5,
    exceptions_dict={ResourceNotFoundError: [], UnexpectedResourceCountError: []},
)
def _wait_for_unique_ogx_pod(client: DynamicClient, namespace: str) -> Pod:
    pods = list(Pod.get(client=client, namespace=namespace, label_selector=OGX_CORE_POD_FILTER))
    if not pods:
        raise ResourceNotFoundError(f"No pods found with label selector {OGX_CORE_POD_FILTER} in namespace {namespace}")
    if len(pods) != 1:
        raise UnexpectedResourceCountError(
            f"Expected exactly 1 pod with label selector {OGX_CORE_POD_FILTER} "
            f"in namespace {namespace}, found {len(pods)}"
        )
    return pods[0]


@retry(wait_timeout=90, sleep=5)
def _wait_for_ogx_client_ready(client: OgxClient) -> bool:
    try:
        client.inspect.health()
        version = client.inspect.version()
        models = client.models.list()
        vector_stores = client.vector_stores.list()
        files = client.files.list()
        LOGGER.info(
            f"OGX server is available! "
            f"(version:{version.version} "
            f"models:{len(models.data)} "
            f"vector_stores:{len(vector_stores.data)} "
            f"files:{len(files.data)})"
        )
    except (APIConnectionError, InternalServerError) as error:
        LOGGER.debug(f"OGX server not ready yet: {error}")
        return False
    except Exception as e:  # noqa: BLE001
        LOGGER.warning(f"Unexpected error checking OGX readiness: {e}")
        return False
    else:
        return True


@pytest.fixture(scope="session", autouse=True)
def _validate_autorag_env() -> None:
    if AUTORAG_PIPELINE_YAML:
        LOGGER.info("AUTORAG_PIPELINE_YAML is set — using legacy YAML upload mode")
    else:
        LOGGER.info("AUTORAG_PIPELINE_YAML is not set — using managed pipeline mode")
    missing = [f"  {var}: {desc}" for var, desc in _AUTORAG_REQUIRED_ENV.items() if not os.environ.get(var)]
    if missing:
        pytest.fail("AutoRAG smoke test requires environment variables:\n" + "\n".join(missing))


# ---------------------------------------------------------------------------
# Override parent namespace fixture with a shorter name to stay under the
# 63-char DNS label limit for KServe predictor hostnames.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def pipelines_namespace(admin_client: DynamicClient) -> Generator[Namespace, Any, Any]:  # noqa: UFN001
    with create_ns(
        admin_client=admin_client,
        name=f"autorag-aqa-{uuid.uuid4().hex[:8]}",
    ) as namespace:
        yield namespace


# ---------------------------------------------------------------------------
# Step 1: Deploy vLLM models (inference + embedding)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def autorag_hf_token_secret(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
) -> Generator[Secret, Any, Any]:
    """HuggingFace token secret for downloading models from hf:// URIs.

    The managed pipeline controller may auto-create this secret, so
    reuse it if it already exists.
    """
    existing_secret = Secret(
        client=admin_client,
        namespace=pipelines_namespace.name,
        name="hf-token-secret",
    )
    if existing_secret.exists:
        yield existing_secret
    else:
        hf_token = os.environ.get("HF_TOKEN", "")
        with Secret(
            client=admin_client,
            namespace=pipelines_namespace.name,
            name="hf-token-secret",
            type="Opaque",
            string_data={"token": hf_token},
        ) as new_secret:
            yield new_secret


@pytest.fixture(scope="class")
def autorag_model_service_account(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    autorag_hf_token_secret: Secret,
) -> Generator[Any, Any, Any]:
    """ServiceAccount with HF token secret for KServe storage initializer."""
    with ServiceAccount(
        client=admin_client,
        namespace=pipelines_namespace.name,
        name="autorag-model-sa",
        secrets=[{"name": autorag_hf_token_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def autorag_inference_runtime(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
) -> Generator[ServingRuntimeFromTemplate, Any, Any]:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="autorag-vllm-inference",
        namespace=pipelines_namespace.name,
        template_name="vllm-cpu-runtime-template",  # that need to be changed when #RHOAIENG-68247 will be fixed
        multi_model=False,
        enable_http=True,
        enable_grpc=False,
    ) as runtime:
        yield runtime


@pytest.fixture(scope="class")
def autorag_inference_service(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    autorag_inference_runtime: ServingRuntimeFromTemplate,
    autorag_model_service_account: Any,
) -> Generator[InferenceService, Any, Any]:
    served_model_name = AUTORAG_LLAMA_STACK_INFERENCE_MODEL_ID or AUTORAG_INFERENCE_MODEL_NAME
    with create_isvc(
        client=admin_client,
        name="autorag-inference",
        namespace=pipelines_namespace.name,
        model_format="vLLM",
        runtime=autorag_inference_runtime.name,
        storage_uri=AUTORAG_INFERENCE_MODEL_URI,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        wait=True,
        timeout=1800,
        model_service_account=autorag_model_service_account.name,
        resources={
            "requests": {"cpu": "2", "memory": "4Gi"},
            "limits": {"cpu": "4", "memory": "8Gi"},
        },
        model_env_variables=[{"name": "VLLM_CPU_KVCACHE_SPACE", "value": "2"}],
        argument=["--served-model-name", served_model_name, "--max-model-len", "4096"],
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def autorag_inference_url(autorag_inference_service: InferenceService) -> str:
    url = autorag_inference_service.instance.status.address.url
    assert url, f"InferenceService {autorag_inference_service.name} has no status.address.url"
    return f"{url}/v1"


@pytest.fixture(scope="class")
def autorag_embedding_runtime(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
) -> Generator[ServingRuntimeFromTemplate, Any, Any]:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="autorag-vllm-embedding",
        namespace=pipelines_namespace.name,
        template_name=RuntimeTemplates.VLLM_CPU_x86,
        multi_model=False,
        enable_http=True,
        enable_grpc=False,
    ) as runtime:
        yield runtime


@pytest.fixture(scope="class")
def autorag_embedding_service(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    autorag_embedding_runtime: ServingRuntimeFromTemplate,
    autorag_model_service_account: Any,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name="autorag-embedding",
        namespace=pipelines_namespace.name,
        model_format="vLLM",
        runtime=autorag_embedding_runtime.name,
        storage_uri=AUTORAG_EMBEDDING_MODEL_URI,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        wait=True,
        timeout=1800,
        model_service_account=autorag_model_service_account.name,
        resources={
            "requests": {"cpu": "2", "memory": "4Gi"},
            "limits": {"cpu": "4", "memory": "8Gi"},
        },
        model_env_variables=[
            {"name": "VLLM_CPU_KVCACHE_SPACE", "value": "2"},
            {"name": "VLLM_ENGINE_ITERATION_TIMEOUT_S", "value": "600"},
            {"name": "VLLM_MAX_NUM_SEQS", "value": "2"},
            {"name": "VLLM_TARGET_DEVICE", "value": "cpu"},
        ],
        argument=[
            "--served-model-name",
            AUTORAG_EMBEDDING_MODEL_NAME,
            "--runner",
            "pooling",
            "--max-model-len",
            AUTORAG_EMBEDDING_MAX_MODEL_LEN,
        ],
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def autorag_embedding_url(autorag_embedding_service: InferenceService) -> str:
    url = autorag_embedding_service.instance.status.address.url
    assert url, f"InferenceService {autorag_embedding_service.name} has no status.address.url"
    return f"{url}/v1"


# ---------------------------------------------------------------------------
# Step 2: Deploy OGX server (formerly LlamaStack)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def autorag_ogx_operator(
    dsc_resource: DataScienceCluster,
) -> Generator[DataScienceCluster, Any, Any]:
    with update_components_in_dsc(
        dsc=dsc_resource,
        components={DscComponents.OGX: DscComponents.ManagementState.MANAGED},
        wait_for_components_state=True,
    ) as dsc:
        yield dsc


@pytest.fixture(scope="class")
def autorag_run_suffix() -> str:
    return uuid.uuid4().hex[:8]


@pytest.fixture(scope="class")
def autorag_ogx_secret(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    autorag_run_suffix: str,
) -> Generator[Secret, Any, Any]:
    with Secret(
        client=admin_client,
        namespace=pipelines_namespace.name,
        name=f"{AUTORAG_RESOURCE_PREFIX}-ogx-secret-{autorag_run_suffix}",
        type="Opaque",
        string_data=AUTORAG_OGX_SECRET_DATA,
    ) as secret:
        yield secret


def _get_postgres_template(secret_name: str, app_label: str) -> dict[str, Any]:
    return {
        "metadata": {"labels": {"app": app_label, "autorag-component": "postgres"}},
        "spec": {
            "containers": [
                {
                    "name": "postgres",
                    "image": POSTGRES_IMAGE,
                    "ports": [{"containerPort": 5432}],
                    "env": [
                        {"name": "POSTGRESQL_DATABASE", "value": "ps_db"},
                        {
                            "name": "POSTGRESQL_USER",
                            "valueFrom": {"secretKeyRef": {"name": secret_name, "key": "postgres-user"}},
                        },
                        {
                            "name": "POSTGRESQL_PASSWORD",
                            "valueFrom": {"secretKeyRef": {"name": secret_name, "key": "postgres-password"}},
                        },
                    ],
                    "volumeMounts": [{"name": "postgresdata", "mountPath": "/var/lib/pgsql/data"}],
                },
            ],
            "volumes": [{"name": "postgresdata", "emptyDir": {}}],
        },
    }


@pytest.fixture(scope="class")
def autorag_postgres_deployment(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    autorag_run_suffix: str,
    autorag_ogx_secret: Secret,
) -> Generator[Deployment, Any, Any]:
    app_label = f"{AUTORAG_RESOURCE_PREFIX}-pg-{autorag_run_suffix}"
    with Deployment(
        client=admin_client,
        namespace=pipelines_namespace.name,
        name=f"{AUTORAG_RESOURCE_PREFIX}-pg-{autorag_run_suffix}",
        min_ready_seconds=5,
        replicas=1,
        selector={"matchLabels": {"app": app_label}},
        strategy={"type": "Recreate"},
        template=_get_postgres_template(secret_name=autorag_ogx_secret.name, app_label=app_label),
    ) as deployment:
        deployment.wait_for_replicas(deployed=True, timeout=240)
        yield deployment


@pytest.fixture(scope="class")
def autorag_postgres_service(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    autorag_run_suffix: str,
    autorag_postgres_deployment: Deployment,
) -> Generator[Service, Any, Any]:
    app_label = f"{AUTORAG_RESOURCE_PREFIX}-pg-{autorag_run_suffix}"
    with Service(
        client=admin_client,
        namespace=pipelines_namespace.name,
        name=f"{AUTORAG_RESOURCE_PREFIX}-pg-{autorag_run_suffix}",
        ports=[{"port": 5432, "targetPort": 5432}],
        selector={"app": app_label},
        wait_for_resource=True,
    ) as service:
        yield service


# ---------------------------------------------------------------------------
# etcd + Milvus (required by OGX milvus-remote vector_io provider)
# ---------------------------------------------------------------------------


def _get_etcd_template(etcd_service_name: str) -> dict[str, Any]:
    template = get_etcd_deployment_template()
    container = template["spec"]["containers"][0]
    container["command"] = [
        "etcd",
        f"--advertise-client-urls=http://{etcd_service_name}:2379",
        "--listen-client-urls=http://0.0.0.0:2379",
        "--data-dir=/etcd",
    ]
    return template


def _get_milvus_template(etcd_service_name: str) -> dict[str, Any]:
    template = get_milvus_deployment_template()
    container = template["spec"]["containers"][0]
    for env in container["env"]:
        if env["name"] == "ETCD_ENDPOINTS":
            env["value"] = f"{etcd_service_name}:2379"
    return template


@pytest.fixture(scope="class")
def autorag_etcd_deployment(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    autorag_run_suffix: str,
) -> Generator[Deployment, Any, Any]:
    app_label = f"{AUTORAG_RESOURCE_PREFIX}-etcd-{autorag_run_suffix}"
    service_name = f"{AUTORAG_RESOURCE_PREFIX}-etcd-{autorag_run_suffix}"
    template = _get_etcd_template(etcd_service_name=service_name)
    template["metadata"]["labels"]["app"] = app_label
    with Deployment(
        client=admin_client,
        namespace=pipelines_namespace.name,
        name=f"{AUTORAG_RESOURCE_PREFIX}-etcd-{autorag_run_suffix}",
        replicas=1,
        selector={"matchLabels": {"app": app_label}},
        strategy={"type": "Recreate"},
        template=template,
    ) as deployment:
        deployment.wait_for_replicas(deployed=True, timeout=120)
        yield deployment


@pytest.fixture(scope="class")
def autorag_etcd_service(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    autorag_run_suffix: str,
    autorag_etcd_deployment: Deployment,
) -> Generator[Service, Any, Any]:
    app_label = f"{AUTORAG_RESOURCE_PREFIX}-etcd-{autorag_run_suffix}"
    with Service(
        client=admin_client,
        namespace=pipelines_namespace.name,
        name=f"{AUTORAG_RESOURCE_PREFIX}-etcd-{autorag_run_suffix}",
        ports=[{"port": 2379, "targetPort": 2379}],
        selector={"app": app_label},
        wait_for_resource=True,
    ) as service:
        yield service


@pytest.fixture(scope="class")
def autorag_milvus_deployment(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    autorag_run_suffix: str,
    autorag_etcd_deployment: Deployment,
    autorag_etcd_service: Service,
) -> Generator[Deployment, Any, Any]:
    app_label = f"{AUTORAG_RESOURCE_PREFIX}-milvus-{autorag_run_suffix}"
    etcd_service_name = f"{AUTORAG_RESOURCE_PREFIX}-etcd-{autorag_run_suffix}"
    template = _get_milvus_template(etcd_service_name=etcd_service_name)
    template["metadata"]["labels"]["app"] = app_label
    with Deployment(
        client=admin_client,
        namespace=pipelines_namespace.name,
        name=f"{AUTORAG_RESOURCE_PREFIX}-milvus-{autorag_run_suffix}",
        min_ready_seconds=5,
        replicas=1,
        selector={"matchLabels": {"app": app_label}},
        strategy={"type": "Recreate"},
        template=template,
    ) as deployment:
        deployment.wait_for_replicas(deployed=True, timeout=240)
        yield deployment


@pytest.fixture(scope="class")
def autorag_milvus_service(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    autorag_run_suffix: str,
    autorag_milvus_deployment: Deployment,
) -> Generator[Service, Any, Any]:
    app_label = f"{AUTORAG_RESOURCE_PREFIX}-milvus-{autorag_run_suffix}"
    with Service(
        client=admin_client,
        namespace=pipelines_namespace.name,
        name=f"{AUTORAG_RESOURCE_PREFIX}-milvus-{autorag_run_suffix}",
        ports=[{"name": "grpc", "port": 19530, "targetPort": 19530}],
        selector={"app": app_label},
        wait_for_resource=True,
    ) as service:
        yield service


@pytest.fixture(scope="class")
def autorag_ogx_server(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    autorag_ogx_operator: DataScienceCluster,
    autorag_ogx_secret: Secret,
    autorag_postgres_deployment: Deployment,
    autorag_postgres_service: Service,
    autorag_milvus_service: Service,
    autorag_inference_url: str,
    autorag_embedding_url: str,
    autorag_inference_route: Route,
) -> Generator[OgxServer, Any, Any]:
    inference_catalog_model_id = AUTORAG_LLAMA_STACK_INFERENCE_MODEL_ID or AUTORAG_INFERENCE_MODEL_NAME

    _wait_for_vllm_model_ready(
        vllm_base_url=f"https://{autorag_inference_route.host}/v1",
        model_name=inference_catalog_model_id,
    )

    secret_name = autorag_ogx_secret.name
    postgres_service_name = autorag_postgres_service.name

    env_vars = [
        {"name": "INFERENCE_MODEL", "value": inference_catalog_model_id},
        {"name": "INFERENCE_PROVIDER_MODEL_ID", "value": inference_catalog_model_id},
        {
            "name": "VLLM_API_TOKEN",
            "valueFrom": {"secretKeyRef": {"name": secret_name, "key": "vllm-api-token"}},
        },
        {"name": "VLLM_URL", "value": autorag_inference_url},
        {"name": "VLLM_TLS_VERIFY", "value": "false"},
        {"name": "VLLM_MAX_TOKENS", "value": "128"},
        {"name": "FMS_ORCHESTRATOR_URL", "value": "http://localhost"},
        {"name": "EMBEDDING_MODEL", "value": AUTORAG_EMBEDDING_MODEL_NAME},
        {"name": "EMBEDDING_PROVIDER_MODEL_ID", "value": AUTORAG_EMBEDDING_MODEL_NAME},
        {"name": "VLLM_EMBEDDING_URL", "value": autorag_embedding_url},
        {
            "name": "VLLM_EMBEDDING_API_TOKEN",
            "valueFrom": {"secretKeyRef": {"name": secret_name, "key": "vllm-embedding-api-token"}},
        },
        {"name": "VLLM_EMBEDDING_MAX_TOKENS", "value": "768"},
        {"name": "VLLM_EMBEDDING_TLS_VERIFY", "value": "false"},
        {"name": "POSTGRES_HOST", "value": postgres_service_name},
        {"name": "POSTGRES_PORT", "value": "5432"},
        {
            "name": "POSTGRES_USER",
            "valueFrom": {"secretKeyRef": {"name": secret_name, "key": "postgres-user"}},
        },
        {
            "name": "POSTGRES_PASSWORD",
            "valueFrom": {"secretKeyRef": {"name": secret_name, "key": "postgres-password"}},
        },
        {"name": "POSTGRES_DB", "value": "ps_db"},
        {"name": "POSTGRES_TABLE_NAME", "value": "llamastack_kvstore"},
        {"name": "MILVUS_ENDPOINT", "value": f"http://{autorag_milvus_service.name}:19530"},
        {
            "name": "MILVUS_TOKEN",
            "valueFrom": {"secretKeyRef": {"name": secret_name, "key": "milvus-token"}},
        },
        {"name": "MILVUS_CONSISTENCY_LEVEL", "value": "Bounded"},
    ]

    ogx_config: dict[str, Any] = {
        "distribution": {"name": "rh-dev"},
        "workload": {
            "resources": {
                "requests": {"cpu": "1", "memory": "2Gi"},
                "limits": {"cpu": "2", "memory": "4Gi"},
            },
            "overrides": {
                "env": env_vars,
            },
        },
    }

    name = generate_random_name(prefix="autorag-ogx")
    with _create_ogx_server(
        client=admin_client,
        name=name,
        namespace=pipelines_namespace.name,
        config=ogx_config,
    ) as ogx_srv:
        ogx_srv.wait_for_status(status=OgxServer.Status.READY, timeout=900)
        yield ogx_srv


@pytest.fixture(scope="class")
def autorag_ogx_deployment(
    admin_client: DynamicClient,
    autorag_ogx_server: OgxServer,
) -> Deployment:
    deployment = Deployment(
        client=admin_client,
        namespace=autorag_ogx_server.namespace,
        name=autorag_ogx_server.name,
        min_ready_seconds=10,
    )
    deployment.timeout_seconds = 240
    deployment.wait(timeout=240)
    deployment.wait_for_replicas()
    _wait_for_unique_ogx_pod(client=admin_client, namespace=autorag_ogx_server.namespace)

    return deployment


@pytest.fixture(scope="class")
def autorag_inference_route(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    autorag_inference_service: InferenceService,
) -> Generator[Route, Any, Any]:
    route_name = generate_random_name(prefix="autorag-inf", length=12)
    with Route(
        client=admin_client,
        namespace=pipelines_namespace.name,
        name=route_name,
        service=f"{autorag_inference_service.name}-predictor",
        wait_for_resource=True,
    ) as route:
        ResourceEditor(
            patches={
                route: {
                    "spec": {
                        "tls": {
                            "termination": "edge",
                            "insecureEdgeTerminationPolicy": "Redirect",
                        }
                    }
                }
            }
        ).update()
        route.wait(timeout=60)
        yield route


@pytest.fixture(scope="class")
def autorag_ogx_route(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    autorag_ogx_deployment: Deployment,
) -> Generator[Route, Any, Any]:
    route_name = generate_random_name(prefix="autorag-ogx", length=12)
    with Route(
        client=admin_client,
        namespace=pipelines_namespace.name,
        name=route_name,
        service=f"{autorag_ogx_deployment.name}-service",
        wait_for_resource=True,
    ) as route:
        ResourceEditor(
            patches={
                route: {
                    "spec": {
                        "tls": {
                            "termination": "edge",
                            "insecureEdgeTerminationPolicy": "Redirect",
                        }
                    },
                    "metadata": {
                        "annotations": {Annotations.HaproxyRouterOpenshiftIo.TIMEOUT: "10m"},
                    },
                }
            }
        ).update()
        route.wait(timeout=60)
        yield route


# ---------------------------------------------------------------------------
# Step 3: Connect to OGX server and discover models
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def autorag_ogx_client(
    autorag_ogx_route: Route,
) -> Generator[OgxClient, Any, Any]:
    http_client = httpx.Client(verify=OGX_CLIENT_VERIFY_SSL, timeout=300)
    try:
        client = OgxClient(
            base_url=f"https://{autorag_ogx_route.host}",
            max_retries=3,
            http_client=http_client,
            timeout=300,
        )
        _wait_for_ogx_client_ready(client=client)
        yield client
    finally:
        http_client.close()


@pytest.fixture(scope="class")
def autorag_ogx_url(autorag_ogx_route: Route) -> str:
    return f"https://{autorag_ogx_route.host}"


def _resolve_model_id(registered_ids: set[str], model_name: str) -> str | None:
    if model_name in registered_ids:
        return model_name
    matches = [mid for mid in registered_ids if mid.endswith(f"/{model_name}")]
    return matches[0] if len(matches) == 1 else None


def _log_registered_models(client: OgxClient) -> set[str]:
    models = client.models.list()
    registered_ids = {model.id for model in models.data}
    LOGGER.info(
        "OGX registered models",
        models=[
            {
                "id": model.id,
                "model_type": str(getattr(model, "model_type", "?")),
                "custom_metadata": getattr(model, "custom_metadata", {}),
            }
            for model in models.data
        ],
    )
    return registered_ids


def _wait_for_vllm_model_ready(vllm_base_url: str, model_name: str, timeout: int = 300) -> None:

    LOGGER.info("Probing vLLM reachability from test runner", url=vllm_base_url, model=model_name)
    try:
        with httpx.Client(verify=False, timeout=5) as probe:
            probe.get(f"{vllm_base_url}/models")
    except Exception as exc:  # noqa: BLE001
        LOGGER.info(
            "vLLM URL not reachable from test runner (cluster-internal); skipping readiness wait",
            url=vllm_base_url,
            model=model_name,
            reason=str(exc),
        )
        return

    def _check_model() -> bool:
        with httpx.Client(verify=False, timeout=30) as http_client:
            resp = http_client.get(f"{vllm_base_url}/models")
            if resp.status_code == 200:
                model_ids = {model.get("id", "") for model in resp.json().get("data", [])}
                LOGGER.info("vLLM models", url=vllm_base_url, models=sorted(model_ids))
                if _resolve_model_id(model_ids, model_name) is not None:
                    LOGGER.info("vLLM model is ready", model=model_name)
                    return True
            else:
                LOGGER.debug("vLLM /v1/models returned non-200", status=resp.status_code)
        return False

    LOGGER.info("vLLM URL is reachable; waiting for model", url=vllm_base_url, model=model_name)
    for ready in TimeoutSampler(
        wait_timeout=timeout,
        sleep=15,
        func=_check_model,
        exceptions_dict={httpx.HTTPError: [], ConnectionError: [], OSError: []},
    ):
        if ready:
            return

    raise TimeoutError(
        f"vLLM did not serve model '{model_name}' at '{vllm_base_url}' within {timeout}s. "
        f"Verify --served-model-name is set correctly in the ISVC spec."
    )


@pytest.fixture(scope="class")
def autorag_discovered_models(
    autorag_ogx_client: OgxClient,
) -> tuple[str, str]:
    registered_ids = _log_registered_models(client=autorag_ogx_client)

    embedding_id = _resolve_model_id(registered_ids=registered_ids, model_name=AUTORAG_EMBEDDING_MODEL_NAME)
    assert embedding_id is not None, (
        f"Embedding model '{AUTORAG_EMBEDDING_MODEL_NAME}' not registered in OGX server. "
        f"Available: {sorted(registered_ids)}"
    )

    generation_id = _resolve_model_id(registered_ids=registered_ids, model_name=AUTORAG_INFERENCE_MODEL_NAME)
    if generation_id is None and AUTORAG_LLAMA_STACK_INFERENCE_MODEL_ID:
        generation_id = _resolve_model_id(
            registered_ids=registered_ids, model_name=AUTORAG_LLAMA_STACK_INFERENCE_MODEL_ID
        )
    assert generation_id is not None, (
        f"Generation model not registered in OGX server. "
        f"Looked for '{AUTORAG_INFERENCE_MODEL_NAME}'"
        + (f" and '{AUTORAG_LLAMA_STACK_INFERENCE_MODEL_ID}'" if AUTORAG_LLAMA_STACK_INFERENCE_MODEL_ID else "")
        + f". Available: {sorted(registered_ids)}\n"
        f"The rh-dev distribution validates INFERENCE_MODEL against its model catalog. "
        f"If '{AUTORAG_INFERENCE_MODEL_NAME}' is not a catalog model (e.g. a Qwen or custom model), "
        f"set AUTORAG_LLAMA_STACK_INFERENCE_MODEL_ID to a supported catalog name such as "
        f"'meta-llama/Llama-3.2-1B-Instruct'. Both the vLLM --served-model-name and INFERENCE_MODEL "
        f"will use that catalog name; the actual weights are loaded from AUTORAG_INFERENCE_MODEL_URI."
    )

    LOGGER.info("Using models", embedding=embedding_id, generation=generation_id)
    return embedding_id, generation_id


# ---------------------------------------------------------------------------
# Step 4: DSPA / pipeline fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def autorag_managed_pipeline(
    dspa: DataSciencePipelinesApplication,
    dspa_api_url: str,
    dspa_auth_headers: dict[str, str],
    dspa_ca_bundle_file: str,
) -> dict[str, str] | None:
    """Discovered managed pipeline info, or None in legacy mode."""
    if not use_managed_pipelines(yaml_env_value=AUTORAG_PIPELINE_YAML):
        return None
    return wait_for_managed_pipeline(
        api_url=dspa_api_url,
        headers=dspa_auth_headers,
        display_name=MANAGED_PIPELINE_AUTORAG,
        ca_bundle=dspa_ca_bundle_file,
        timeout=DSPA_READY_BUFFER_SECONDS + MANAGED_PIPELINE_WAIT_TIMEOUT,
        poll_interval=MANAGED_PIPELINE_POLL_INTERVAL,
    )


@pytest.fixture(scope="class")
def autorag_pipeline_yaml_path() -> str | None:
    """Resolve the AutoRAG pipeline YAML. None in managed mode."""
    if not AUTORAG_PIPELINE_YAML:
        return None
    return resolve_pipeline_yaml(value=AUTORAG_PIPELINE_YAML)


@pytest.fixture(scope="class")
def autorag_test_data(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    dspa_s3_credentials: Secret,
) -> None:
    """Download AutoRAG test data from external S3 and upload to built-in MinIO."""
    src_bucket = shlex.quote(s=AUTORAG_S3_BUCKET)
    src_input_key = shlex.quote(s=AUTORAG_INPUT_DATA_KEY)
    src_test_key = shlex.quote(s=AUTORAG_TEST_DATA_KEY)
    dst_bucket = shlex.quote(s=DSPA_S3_BUCKET)

    minio_endpoint = f"http://minio-{DSPA_NAME}.{pipelines_namespace.name}.svc.cluster.local:9000"

    mc_setup = (
        "export MC_CONFIG_DIR=/work/.mc && "
        "mc alias set src $SRC_ENDPOINT $SRC_ACCESS_KEY $SRC_SECRET_KEY && "
        "mc alias set dst $DST_ENDPOINT $DST_ACCESS_KEY $DST_SECRET_KEY"
    )
    mc_copy = (
        f"mc cp --recursive src/{src_bucket}/{src_input_key} /work/input_data/ && "
        f"mc cp src/{src_bucket}/{src_test_key} /work/benchmark_data.json && "
        f"mc mb --ignore-existing dst/{dst_bucket} && "
        f"mc cp --recursive /work/input_data/ dst/{dst_bucket}/{src_input_key}/ && "
        f"mc cp /work/benchmark_data.json dst/{dst_bucket}/{src_test_key}"
    )

    src_endpoint = os.environ.get("AWS_S3_ENDPOINT", "https://s3.amazonaws.com")
    src_access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    src_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")

    pod_name = f"autorag-data-uploader-{uuid.uuid4().hex[:8]}"
    with Pod(
        client=admin_client,
        name=pod_name,
        namespace=pipelines_namespace.name,
        restart_policy="Never",
        volumes=[{"name": "work", "emptyDir": {}}],
        containers=[
            {
                "name": "minio-uploader",
                "image": MINIO_MC_IMAGE,
                "command": ["/bin/sh", "-c"],
                "args": [f"{mc_setup} && {mc_copy}"],
                "volumeMounts": [{"name": "work", "mountPath": "/work"}],
                "securityContext": MINIO_UPLOADER_SECURITY_CONTEXT,
                "env": [
                    {"name": "SRC_ENDPOINT", "value": src_endpoint},
                    {"name": "SRC_ACCESS_KEY", "value": src_access_key},
                    {"name": "SRC_SECRET_KEY", "value": src_secret_key},
                    {"name": "DST_ENDPOINT", "value": minio_endpoint},
                    {
                        "name": "DST_ACCESS_KEY",
                        "valueFrom": {"secretKeyRef": {"name": DSPA_S3_SECRET, "key": "accesskey"}},
                    },
                    {
                        "name": "DST_SECRET_KEY",
                        "valueFrom": {"secretKeyRef": {"name": DSPA_S3_SECRET, "key": "secretkey"}},
                    },
                ],
            }
        ],
        wait_for_resource=True,
    ) as upload_pod:
        try:
            upload_pod.wait_for_status(status="Succeeded", timeout=300)
        except TimeoutExpiredError:
            try:
                LOGGER.error("Data upload pod logs", logs=upload_pod.log())
            except Exception:  # noqa: BLE001
                LOGGER.warning("Could not fetch upload pod logs")
            raise

    LOGGER.info("AutoRAG test data uploaded to MinIO")


@pytest.fixture(scope="class")
def autorag_ogx_url_secret(
    admin_client: DynamicClient,
    autorag_run_suffix: str,
    autorag_ogx_url: str,
    pipelines_namespace: Namespace,
) -> Generator[Secret, Any, Any]:
    with Secret(
        client=admin_client,
        name=f"{AUTORAG_RESOURCE_PREFIX}-ogx-url-{autorag_run_suffix}",  # pragma: allowlist secret
        namespace=pipelines_namespace.name,
        string_data={
            "OGX_CLIENT_BASE_URL": autorag_ogx_url,
            "OGX_CLIENT_API_KEY": "unused",  # pragma: allowlist secret
        },
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def autorag_pipeline_id(
    dspa_api_url: str,
    dspa_auth_headers: dict[str, str],
    dspa_ca_bundle_file: str,
    autorag_pipeline_yaml_path: str | None,
    autorag_managed_pipeline: dict[str, str] | None,
) -> Generator[str, Any, Any]:
    """Pipeline ID — from managed discovery or YAML upload."""
    if autorag_managed_pipeline is not None:
        yield autorag_managed_pipeline["pipeline_id"]
    else:
        assert autorag_pipeline_yaml_path is not None, "AUTORAG_PIPELINE_YAML must be set for legacy mode"
        run_suffix = uuid.uuid4().hex[:8]

        @retry(wait_timeout=120, sleep=10, exceptions_dict={requests.HTTPError: [], requests.ConnectionError: []})
        def _upload() -> str:
            return upload_pipeline(
                api_url=dspa_api_url,
                headers=dspa_auth_headers,
                pipeline_yaml_path=autorag_pipeline_yaml_path,
                pipeline_name=f"autorag-smoke-{run_suffix}",
                ca_bundle=dspa_ca_bundle_file,
            )

        pipeline_id = _upload()
        yield pipeline_id
        delete_pipeline(
            api_url=dspa_api_url,
            headers=dspa_auth_headers,
            pipeline_id=pipeline_id,
            ca_bundle=dspa_ca_bundle_file,
        )


@pytest.fixture(scope="class")
def autorag_run_id(
    dspa_api_url: str,
    dspa_auth_headers: dict[str, str],
    dspa_ca_bundle_file: str,
    autorag_pipeline_id: str,
    autorag_managed_pipeline: dict[str, str] | None,
    autorag_ogx_url_secret: Secret,
    autorag_discovered_models: tuple[str, str],
    autorag_inference_url: str,
    autorag_embedding_url: str,
    dspa_s3_credentials: Secret,
    autorag_test_data: None,
) -> Generator[str, Any, Any]:
    embedding_model, generation_model = autorag_discovered_models

    parameters: dict[str, Any] = {
        "input_data_secret_name": dspa_s3_credentials.name,
        "input_data_bucket_name": DSPA_S3_BUCKET,
        "input_data_key": AUTORAG_INPUT_DATA_KEY,
        "test_data_secret_name": dspa_s3_credentials.name,
        "test_data_bucket_name": DSPA_S3_BUCKET,
        "test_data_key": AUTORAG_TEST_DATA_KEY,
        "ogx_secret_name": autorag_ogx_url_secret.name,
        "optimization_max_rag_patterns": AUTORAG_MAX_RAG_PATTERNS,
        "optimization_metric": AUTORAG_OPTIMIZATION_METRIC,
        "embedding_models": [embedding_model],
        "generation_models": [generation_model],
        "vector_io_provider_id": "milvus-remote",
    }

    if autorag_managed_pipeline is not None:
        run_id = create_pipeline_run_managed(
            api_url=dspa_api_url,
            headers=dspa_auth_headers,
            pipeline_id=autorag_managed_pipeline["pipeline_id"],
            pipeline_version_id=autorag_managed_pipeline["pipeline_version_id"],
            run_name=f"autorag-smoke-{uuid.uuid4().hex[:8]}",
            parameters=parameters,
            ca_bundle=dspa_ca_bundle_file,
        )
    else:
        run_id = create_pipeline_run(
            api_url=dspa_api_url,
            headers=dspa_auth_headers,
            pipeline_id=autorag_pipeline_id,
            run_name=f"autorag-smoke-{uuid.uuid4().hex[:8]}",
            parameters=parameters,
            ca_bundle=dspa_ca_bundle_file,
        )

    yield run_id
    delete_pipeline_run(
        api_url=dspa_api_url,
        headers=dspa_auth_headers,
        run_id=run_id,
        ca_bundle=dspa_ca_bundle_file,
    )
