import os
import uuid
from collections.abc import Generator
from typing import Any

import httpx
import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from llama_stack_client import LlamaStackClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.data_science_pipelines_application import DataSciencePipelinesApplication
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service

from tests.llama_stack.constants import (
    LLAMA_STACK_DISTRIBUTION_SECRET_DATA,
    LLS_CLIENT_VERIFY_SSL,
    POSTGRES_IMAGE,
)
from tests.llama_stack.utils import (
    create_llama_stack_distribution,
    wait_for_llama_stack_client_ready,
    wait_for_unique_llama_stack_pod,
)
from tests.pipelines_components.constants import (
    AUTORAG_DSPA_NAME,
    AUTORAG_DSPA_NAMESPACE,
    AUTORAG_INPUT_DATA_KEY,
    AUTORAG_MAX_RAG_PATTERNS,
    AUTORAG_OPTIMIZATION_METRIC,
    AUTORAG_PIPELINE_YAML,
    AUTORAG_S3_BUCKET,
    AUTORAG_S3_SECRET_NAME,
    AUTORAG_TEST_DATA_KEY,
)
from tests.pipelines_components.utils import (
    create_pipeline_run,
    delete_pipeline,
    delete_pipeline_run,
    resolve_pipeline_yaml,
    upload_pipeline,
)
from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import Annotations, DscComponents, KServeDeploymentType, RuntimeTemplates
from utilities.data_science_cluster_utils import update_components_in_dsc
from utilities.general import generate_random_name
from utilities.inference_utils import create_isvc
from utilities.resources.llama_stack_distribution import LlamaStackDistribution
from utilities.serving_runtime import ServingRuntimeFromTemplate

LOGGER = structlog.get_logger(name=__name__)

AUTORAG_LLAMA_STACK_SECRET_NAME: str = "autorag-llama-stack"
AUTORAG_RESOURCE_PREFIX: str = "autorag-smoke"

# User-provided env vars for the models to deploy
AUTORAG_INFERENCE_MODEL_URI: str = os.environ.get("AUTORAG_INFERENCE_MODEL_URI", "")
AUTORAG_INFERENCE_MODEL_NAME: str = os.environ.get("AUTORAG_INFERENCE_MODEL_NAME", "")
AUTORAG_EMBEDDING_MODEL_URI: str = os.environ.get("AUTORAG_EMBEDDING_MODEL_URI", "")
AUTORAG_EMBEDDING_MODEL_NAME: str = os.environ.get("AUTORAG_EMBEDDING_MODEL_NAME", "")

_AUTORAG_REQUIRED_ENV = {
    "AUTORAG_PIPELINE_YAML": "Path to compiled AutoRAG pipeline YAML",
    "AUTORAG_DSPA_NAMESPACE": "Namespace with pre-existing DSPA",
    "AUTORAG_S3_SECRET_NAME": "S3 credentials secret name in the DSPA namespace",
    "AUTORAG_INFERENCE_MODEL_URI": "Storage URI for inference model (e.g. s3://bucket/model or hf://org/model)",
    "AUTORAG_INFERENCE_MODEL_NAME": "Inference model name (e.g. granite-3b-instruct)",
    "AUTORAG_EMBEDDING_MODEL_URI": "Storage URI for embedding model (e.g. s3://bucket/model or hf://org/model)",
    "AUTORAG_EMBEDDING_MODEL_NAME": "Embedding model name (e.g. bge-m3)",
}


@pytest.fixture(scope="session", autouse=True)
def _validate_autorag_env() -> None:
    missing = [f"  {var}: {desc}" for var, desc in _AUTORAG_REQUIRED_ENV.items() if not os.environ.get(var)]
    if missing:
        pytest.skip("AutoRAG smoke test requires environment variables:\n" + "\n".join(missing))


# ---------------------------------------------------------------------------
# Step 1: Deploy vLLM models (inference + embedding)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def autorag_namespace(admin_client: DynamicClient) -> Namespace:
    ns = Namespace(client=admin_client, name=AUTORAG_DSPA_NAMESPACE)
    assert ns.exists, f"Namespace '{AUTORAG_DSPA_NAMESPACE}' does not exist"
    return ns


@pytest.fixture(scope="class")
def autorag_inference_runtime(
    admin_client: DynamicClient,
    autorag_namespace: Namespace,
) -> Generator[ServingRuntimeFromTemplate, Any, Any]:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="autorag-vllm-inference",
        namespace=autorag_namespace.name,
        template_name=RuntimeTemplates.VLLM_CPU_x86,
        multi_model=False,
        enable_http=True,
        enable_grpc=False,
    ) as runtime:
        yield runtime


@pytest.fixture(scope="class")
def autorag_inference_service(
    admin_client: DynamicClient,
    autorag_namespace: Namespace,
    autorag_inference_runtime: ServingRuntimeFromTemplate,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name="autorag-inference",
        namespace=autorag_namespace.name,
        model_format="vLLM",
        runtime=autorag_inference_runtime.name,
        storage_uri=AUTORAG_INFERENCE_MODEL_URI,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        wait=True,
        resources={
            "requests": {"cpu": "4", "memory": "8Gi"},
            "limits": {"cpu": "8", "memory": "16Gi"},
        },
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def autorag_inference_url(autorag_inference_service: InferenceService) -> str:
    url = autorag_inference_service.instance.status.url
    assert url, "Inference InferenceService has no URL"
    return f"{url}/v1"


@pytest.fixture(scope="class")
def autorag_embedding_runtime(
    admin_client: DynamicClient,
    autorag_namespace: Namespace,
) -> Generator[ServingRuntimeFromTemplate, Any, Any]:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="autorag-vllm-embedding",
        namespace=autorag_namespace.name,
        template_name=RuntimeTemplates.VLLM_CPU_x86,
        multi_model=False,
        enable_http=True,
        enable_grpc=False,
    ) as runtime:
        yield runtime


@pytest.fixture(scope="class")
def autorag_embedding_service(
    admin_client: DynamicClient,
    autorag_namespace: Namespace,
    autorag_embedding_runtime: ServingRuntimeFromTemplate,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name="autorag-embedding",
        namespace=autorag_namespace.name,
        model_format="vLLM",
        runtime=autorag_embedding_runtime.name,
        storage_uri=AUTORAG_EMBEDDING_MODEL_URI,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        wait=True,
        resources={
            "requests": {"cpu": "2", "memory": "4Gi"},
            "limits": {"cpu": "4", "memory": "16Gi"},
        },
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def autorag_embedding_url(autorag_embedding_service: InferenceService) -> str:
    url = autorag_embedding_service.instance.status.url
    assert url, "Embedding InferenceService has no URL"
    return f"{url}/v1"


# ---------------------------------------------------------------------------
# Step 2: Deploy Llama Stack (using patterns from tests/llama_stack)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def autorag_llama_stack_operator(
    dsc_resource: DataScienceCluster,
) -> Generator[DataScienceCluster, Any, Any]:
    with update_components_in_dsc(
        dsc=dsc_resource,
        components={DscComponents.LLAMASTACKOPERATOR: DscComponents.ManagementState.MANAGED},
        wait_for_components_state=True,
    ) as dsc:
        yield dsc


@pytest.fixture(scope="class")
def autorag_run_suffix() -> str:
    return uuid.uuid4().hex[:8]


@pytest.fixture(scope="class")
def autorag_llama_stack_dist_secret(
    admin_client: DynamicClient,
    autorag_namespace: Namespace,
    autorag_run_suffix: str,
) -> Generator[Secret, Any, Any]:
    with Secret(
        client=admin_client,
        namespace=autorag_namespace.name,
        name=f"{AUTORAG_RESOURCE_PREFIX}-ls-secret-{autorag_run_suffix}",
        type="Opaque",
        string_data=LLAMA_STACK_DISTRIBUTION_SECRET_DATA,
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
                            "valueFrom": {
                                "secretKeyRef": {"name": secret_name, "key": "postgres-user"}
                            },
                        },
                        {
                            "name": "POSTGRESQL_PASSWORD",
                            "valueFrom": {
                                "secretKeyRef": {"name": secret_name, "key": "postgres-password"}
                            },
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
    autorag_namespace: Namespace,
    autorag_run_suffix: str,
    autorag_llama_stack_dist_secret: Secret,
) -> Generator[Deployment, Any, Any]:
    app_label = f"{AUTORAG_RESOURCE_PREFIX}-pg-{autorag_run_suffix}"
    with Deployment(
        client=admin_client,
        namespace=autorag_namespace.name,
        name=f"{AUTORAG_RESOURCE_PREFIX}-pg-{autorag_run_suffix}",
        min_ready_seconds=5,
        replicas=1,
        selector={"matchLabels": {"app": app_label}},
        strategy={"type": "Recreate"},
        template=_get_postgres_template(secret_name=autorag_llama_stack_dist_secret.name, app_label=app_label),
    ) as deployment:
        deployment.wait_for_replicas(deployed=True, timeout=240)
        yield deployment


@pytest.fixture(scope="class")
def autorag_postgres_service(
    admin_client: DynamicClient,
    autorag_namespace: Namespace,
    autorag_run_suffix: str,
    autorag_postgres_deployment: Deployment,
) -> Generator[Service, Any, Any]:
    app_label = f"{AUTORAG_RESOURCE_PREFIX}-pg-{autorag_run_suffix}"
    with Service(
        client=admin_client,
        namespace=autorag_namespace.name,
        name=f"{AUTORAG_RESOURCE_PREFIX}-pg-{autorag_run_suffix}",
        ports=[{"port": 5432, "targetPort": 5432}],
        selector={"app": app_label},
        wait_for_resource=True,
    ) as service:
        yield service


@pytest.fixture(scope="class")
def autorag_llama_stack_distribution(
    admin_client: DynamicClient,
    autorag_namespace: Namespace,
    autorag_llama_stack_operator: DataScienceCluster,
    autorag_llama_stack_dist_secret: Secret,
    autorag_postgres_deployment: Deployment,
    autorag_postgres_service: Service,
    autorag_inference_url: str,
    autorag_embedding_url: str,
) -> Generator[LlamaStackDistribution, Any, Any]:
    secret_name = autorag_llama_stack_dist_secret.name
    postgres_service_name = autorag_postgres_service.name

    env_vars = [
        {"name": "INFERENCE_MODEL", "value": AUTORAG_INFERENCE_MODEL_NAME},
        {
            "name": "VLLM_API_TOKEN",
            "valueFrom": {"secretKeyRef": {"name": secret_name, "key": "vllm-api-token"}},
        },
        {"name": "VLLM_URL", "value": autorag_inference_url},
        {"name": "VLLM_TLS_VERIFY", "value": "false"},
        {"name": "VLLM_MAX_TOKENS", "value": "16384"},
        {"name": "FMS_ORCHESTRATOR_URL", "value": "http://localhost"},
        {"name": "EMBEDDING_MODEL", "value": AUTORAG_EMBEDDING_MODEL_NAME},
        {"name": "EMBEDDING_PROVIDER_MODEL_ID", "value": AUTORAG_EMBEDDING_MODEL_NAME},
        {"name": "VLLM_EMBEDDING_URL", "value": autorag_embedding_url},
        {
            "name": "VLLM_EMBEDDING_API_TOKEN",
            "valueFrom": {"secretKeyRef": {"name": secret_name, "key": "vllm-embedding-api-token"}},
        },
        {"name": "VLLM_EMBEDDING_MAX_TOKENS", "value": "8192"},
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
        {"name": "ENABLE_INLINE_MILVUS", "value": "true"},
    ]

    server_config: dict[str, Any] = {
        "containerSpec": {
            "resources": {
                "requests": {"cpu": "2", "memory": "3Gi"},
                "limits": {"cpu": "4", "memory": "6Gi"},
            },
            "env": env_vars,
            "name": "llama-stack",
            "port": 8321,
        },
        "distribution": {"name": "rh-dev"},
    }

    name = generate_random_name(prefix="autorag-llama-stack")
    with create_llama_stack_distribution(
        client=admin_client,
        name=name,
        namespace=autorag_namespace.name,
        replicas=1,
        server=server_config,
    ) as lls_dist:
        lls_dist.wait_for_status(status=LlamaStackDistribution.Status.READY, timeout=600)
        yield lls_dist


@pytest.fixture(scope="class")
def autorag_llama_stack_deployment(
    admin_client: DynamicClient,
    autorag_llama_stack_distribution: LlamaStackDistribution,
) -> Deployment:
    deployment = Deployment(
        client=admin_client,
        namespace=autorag_llama_stack_distribution.namespace,
        name=autorag_llama_stack_distribution.name,
        min_ready_seconds=10,
    )
    deployment.timeout_seconds = 240
    deployment.wait(timeout=240)
    deployment.wait_for_replicas()
    wait_for_unique_llama_stack_pod(
        client=admin_client, namespace=autorag_llama_stack_distribution.namespace
    )
    return deployment


@pytest.fixture(scope="class")
def autorag_llama_stack_route(
    admin_client: DynamicClient,
    autorag_namespace: Namespace,
    autorag_llama_stack_deployment: Deployment,
) -> Generator[Route, Any, Any]:
    route_name = generate_random_name(prefix="autorag-ls", length=12)
    with Route(
        client=admin_client,
        namespace=autorag_namespace.name,
        name=route_name,
        service=f"{autorag_llama_stack_deployment.name}-service",
        wait_for_resource=True,
    ) as route:
        from ocp_resources.resource_editor import ResourceEditor

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
# Step 3: Connect to Llama Stack and discover models
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def autorag_llama_stack_client(
    autorag_llama_stack_route: Route,
) -> Generator[LlamaStackClient, Any, Any]:
    http_client = httpx.Client(verify=LLS_CLIENT_VERIFY_SSL, timeout=300)
    try:
        client = LlamaStackClient(
            base_url=f"https://{autorag_llama_stack_route.host}",
            max_retries=3,
            http_client=http_client,
            timeout=300,
        )
        wait_for_llama_stack_client_ready(client=client)
        yield client
    finally:
        http_client.close()


@pytest.fixture(scope="class")
def autorag_llama_stack_url(autorag_llama_stack_route: Route) -> str:
    return f"https://{autorag_llama_stack_route.host}"


@pytest.fixture(scope="class")
def autorag_discovered_models(
    autorag_llama_stack_client: LlamaStackClient,
) -> tuple[str, str]:
    """Discover embedding and generation model IDs from the deployed Llama Stack."""
    models = autorag_llama_stack_client.models.list()
    embedding_id = None
    generation_id = None
    for model in models:
        model_type = getattr(model, "custom_metadata", {}).get("model_type", "")
        if model_type == "embedding" and not embedding_id:
            embedding_id = model.id
        elif model_type == "llm" and not generation_id:
            generation_id = model.id

    assert embedding_id, "No embedding model found in Llama Stack"
    assert generation_id, "No generation (LLM) model found in Llama Stack"
    LOGGER.info(f"Discovered models: embedding={embedding_id}, generation={generation_id}")
    return embedding_id, generation_id


# ---------------------------------------------------------------------------
# Step 4: DSPA / pipeline fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def autorag_pipeline_yaml_path() -> str:
    return resolve_pipeline_yaml(value=AUTORAG_PIPELINE_YAML)


@pytest.fixture(scope="class")
def autorag_dspa(admin_client: DynamicClient) -> DataSciencePipelinesApplication:
    dspa = DataSciencePipelinesApplication(
        client=admin_client,
        name=AUTORAG_DSPA_NAME,
        namespace=AUTORAG_DSPA_NAMESPACE,
    )
    assert dspa.exists, (
        f"DSPA '{AUTORAG_DSPA_NAME}' not found in namespace '{AUTORAG_DSPA_NAMESPACE}'. "
        f"Ensure the DSPA is deployed before running AutoRAG tests."
    )
    return dspa


@pytest.fixture(scope="class")
def dspa_api_url(autorag_dspa: DataSciencePipelinesApplication) -> str:
    try:
        url = autorag_dspa.instance.status.components.apiServer.externalUrl
    except AttributeError:
        url = None
    assert url, (
        f"DSPA '{AUTORAG_DSPA_NAME}' in namespace '{AUTORAG_DSPA_NAMESPACE}' has no external URL. "
        f"Ensure the DSPA is Ready before running AutoRAG tests."
    )
    return url


@pytest.fixture(scope="class")
def dspa_auth_headers(current_client_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {current_client_token}"}


@pytest.fixture(scope="class")
def dspa_ca_bundle_file(admin_client: DynamicClient) -> str:
    return create_ca_bundle_file(client=admin_client)


@pytest.fixture(scope="class")
def autorag_llama_stack_secret(
    admin_client: DynamicClient,
    autorag_llama_stack_url: str,
) -> Generator[Secret, Any, Any]:
    with Secret(
        client=admin_client,
        name=AUTORAG_LLAMA_STACK_SECRET_NAME,
        namespace=AUTORAG_DSPA_NAMESPACE,
        string_data={
            "LLAMA_STACK_CLIENT_BASE_URL": autorag_llama_stack_url,
            "LLAMA_STACK_CLIENT_API_KEY": "",
        },
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def autorag_pipeline_id(
    dspa_api_url: str,
    dspa_auth_headers: dict[str, str],
    dspa_ca_bundle_file: str,
    autorag_pipeline_yaml_path: str,
) -> Generator[str, Any, Any]:
    run_suffix = uuid.uuid4().hex[:8]
    pipeline_id = upload_pipeline(
        api_url=dspa_api_url,
        headers=dspa_auth_headers,
        pipeline_yaml_path=autorag_pipeline_yaml_path,
        pipeline_name=f"autorag-smoke-{run_suffix}",
        ca_bundle=dspa_ca_bundle_file,
    )
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
    autorag_llama_stack_secret: Secret,
    autorag_discovered_models: tuple[str, str],
) -> Generator[str, Any, Any]:
    embedding_model, generation_model = autorag_discovered_models

    parameters: dict = {
        "input_data_secret_name": AUTORAG_S3_SECRET_NAME,
        "input_data_bucket_name": AUTORAG_S3_BUCKET,
        "input_data_key": AUTORAG_INPUT_DATA_KEY,
        "test_data_secret_name": AUTORAG_S3_SECRET_NAME,
        "test_data_bucket_name": AUTORAG_S3_BUCKET,
        "test_data_key": AUTORAG_TEST_DATA_KEY,
        "llama_stack_secret_name": autorag_llama_stack_secret.name,
        "optimization_max_rag_patterns": AUTORAG_MAX_RAG_PATTERNS,
        "optimization_metric": AUTORAG_OPTIMIZATION_METRIC,
        "embeddings_models": [embedding_model],
        "generation_models": [generation_model],
    }

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
