import os
import tempfile
from typing import Generator, Any, Dict

import portforward
import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from llama_stack_client import LlamaStackClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from ocp_resources.llama_stack_distribution import LlamaStackDistribution
from ocp_resources.namespace import Namespace
from ocp_resources.config_map import ConfigMap
from simple_logger.logger import get_logger
from timeout_sampler import retry

from tests.llama_stack.utils import create_llama_stack_distribution, wait_for_llama_stack_client_ready
from utilities.constants import DscComponents, Timeout
from utilities.data_science_cluster_utils import update_components_in_dsc
from utilities.rag_utils import ModelInfo


LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="class")
def enabled_llama_stack_operator(dsc_resource: DataScienceCluster) -> Generator[DataScienceCluster, Any, Any]:
    with update_components_in_dsc(
        dsc=dsc_resource,
        components={
            DscComponents.LLAMASTACKOPERATOR: DscComponents.ManagementState.MANAGED,
        },
        wait_for_components_state=True,
    ) as dsc:
        yield dsc


@pytest.fixture(scope="class")
def llama_stack_server_config(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Dict[str, Any]:
    fms_orchestrator_url = "http://localhost"
    inference_model = os.getenv("LLS_CORE_INFERENCE_MODEL", "")
    vllm_api_token = os.getenv("LLS_CORE_VLLM_API_TOKEN", "")
    vllm_url = os.getenv("LLS_CORE_VLLM_URL", "")

    if hasattr(request, "param"):
        if request.param.get("fms_orchestrator_url_fixture"):
            fms_orchestrator_url = request.getfixturevalue(argname=request.param.get("fms_orchestrator_url_fixture"))

        # Override env vars with request parameters if provided
        if request.param.get("inference_model"):
            inference_model = request.param.get("inference_model")
        if request.param.get("vllm_api_token"):
            vllm_api_token = request.param.get("vllm_api_token")
        if request.param.get("vllm_url_fixture"):
            vllm_url = request.getfixturevalue(argname=request.param.get("vllm_url_fixture"))

    return {
        "containerSpec": {
            "resources": {
                "requests": {"cpu": "250m", "memory": "500Mi"},
                "limits": {"cpu": "2", "memory": "12Gi"},
            },
            "env": [
                {
                    "name": "VLLM_URL",
                    "value": vllm_url,
                },
                {"name": "VLLM_API_TOKEN", "value": vllm_api_token},
                {
                    "name": "VLLM_TLS_VERIFY",
                    "value": "false",
                },
                {
                    "name": "INFERENCE_MODEL",
                    "value": inference_model,
                },
                {
                    "name": "MILVUS_DB_PATH",
                    "value": "~/.llama/milvus.db",
                },
                {"name": "FMS_ORCHESTRATOR_URL", "value": fms_orchestrator_url},
            ],
            "command": ["/bin/sh", "-c", "llama stack run /etc/llama-stack/run.yaml"],
            "name": "llama-stack",
            "port": 8321,
        },
        "distribution": {"name": "rh-dev"},
        "userConfig": {"configMapName": "rag-llama-stack-config-map"},
        "storage": {
            "size": "20Gi",
        },
    }


@pytest.fixture(scope="class")
def llama_stack_config_map(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[ConfigMap, Any, Any]:
    with ConfigMap(
        client=admin_client,
        namespace=model_namespace.name,
        name="rag-llama-stack-config-map",
        data={
            "run.yaml": """version: 2
image_name: rh
apis:
- agents
- datasetio
- eval
- inference
- safety
- files
- scoring
- telemetry
- tool_runtime
- vector_io
providers:
  inference:
  - provider_id: vllm-inference
    provider_type: remote::vllm
    config:
      url: ${env.VLLM_URL:=http://localhost:8000/v1}
      max_tokens: ${env.VLLM_MAX_TOKENS:=4096}
      api_token: ${env.VLLM_API_TOKEN:=fake}
      tls_verify: ${env.VLLM_TLS_VERIFY:=true}
  - provider_id: sentence-transformers
    provider_type: inline::sentence-transformers
    config: {}
  vector_io:
  - provider_id: milvus
    provider_type: inline::milvus
    config:
      db_path: /opt/app-root/src/.llama/distributions/rh/milvus.db
      kvstore:
        type: sqlite
        namespace: null
        db_path: /opt/app-root/src/.llama/distributions/rh/milvus_registry.db
  files:
  - provider_id: meta-reference-files
    provider_type: inline::localfs
    config:
      storage_dir: /opt/app-root/src/.llama/distributions/rh/files
      metadata_store:
        type: sqlite
        db_path: /opt/app-root/src/.llama/distributions/rh/files_metadata.db
  safety:
    - provider_id: trustyai_fms
      provider_type: remote::trustyai_fms
      config:
        orchestrator_url: ${env.FMS_ORCHESTRATOR_URL:=}
        ssl_cert_path: ${env.FMS_SSL_CERT_PATH:=}
        shields: {}
  agents:
  - provider_id: meta-reference
    provider_type: inline::meta-reference
    config:
      persistence_store:
        type: sqlite
        namespace: null
        db_path: /opt/app-root/src/.llama/distributions/rh/agents_store.db
      responses_store:
        type: sqlite
        db_path: /opt/app-root/src/.llama/distributions/rh/responses_store.db
  eval:
  - provider_id: trustyai_lmeval
    provider_type: remote::trustyai_lmeval
    config:
        use_k8s: True
        base_url: ${env.VLLM_URL:=http://localhost:8000/v1}
  datasetio:
  - provider_id: huggingface
    provider_type: remote::huggingface
    config:
      kvstore:
        type: sqlite
        namespace: null
        db_path: /opt/app-root/src/.llama/distributions/rh/huggingface_datasetio.db
  - provider_id: localfs
    provider_type: inline::localfs
    config:
      kvstore:
        type: sqlite
        namespace: null
        db_path: /opt/app-root/src/.llama/distributions/rh/localfs_datasetio.db
  scoring:
  - provider_id: basic
    provider_type: inline::basic
    config: {}
  - provider_id: llm-as-judge
    provider_type: inline::llm-as-judge
    config: {}
  - provider_id: braintrust
    provider_type: inline::braintrust
    config:
      openai_api_key: ${env.OPENAI_API_KEY:=}
  telemetry:
  - provider_id: meta-reference
    provider_type: inline::meta-reference
    config:
      service_name: "${env.OTEL_SERVICE_NAME:=\u200b}"
      sinks: ${env.TELEMETRY_SINKS:=console,sqlite}
      sqlite_db_path: /opt/app-root/src/.llama/distributions/rh/trace_store.db
      otel_exporter_otlp_endpoint: ${env.OTEL_EXPORTER_OTLP_ENDPOINT:=}
  tool_runtime:
  - provider_id: brave-search
    provider_type: remote::brave-search
    config:
      api_key: ${env.BRAVE_SEARCH_API_KEY:=}
      max_results: 3
  - provider_id: tavily-search
    provider_type: remote::tavily-search
    config:
      api_key: ${env.TAVILY_SEARCH_API_KEY:=}
      max_results: 3
  - provider_id: rag-runtime
    provider_type: inline::rag-runtime
    config: {}
  - provider_id: model-context-protocol
    provider_type: remote::model-context-protocol
    config: {}
metadata_store:
  type: sqlite
  db_path: /opt/app-root/src/.llama/distributions/rh/registry.db
inference_store:
  type: sqlite
  db_path: /opt/app-root/src/.llama/distributions/rh/inference_store.db
models:
- metadata: {}
  model_id: ${env.INFERENCE_MODEL}
  provider_id: vllm-inference
  model_type: llm
- metadata:
    embedding_dimension: 768
  model_id: granite-embedding-125m
  provider_id: sentence-transformers
  provider_model_id: ibm-granite/granite-embedding-125m-english
  model_type: embedding
shields: []
vector_dbs: []
datasets: []
scoring_fns: []
benchmarks: []
tool_groups:
- toolgroup_id: builtin::websearch
  provider_id: tavily-search
- toolgroup_id: builtin::rag
  provider_id: rag-runtime
server:
  port: 8321
external_providers_dir: /opt/app-root/src/.llama/providers.d"""
        },
    ) as config_map:
        yield config_map


@pytest.fixture(scope="class")
def llama_stack_distribution(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    enabled_llama_stack_operator: DataScienceCluster,
    llama_stack_server_config: Dict[str, Any],
    llama_stack_config_map: ConfigMap,
) -> Generator[LlamaStackDistribution, None, None]:
    with create_llama_stack_distribution(
        client=admin_client,
        name="llama-stack-distribution",
        namespace=model_namespace.name,
        replicas=1,
        server=llama_stack_server_config,
    ) as lls_dist:
        lls_dist.wait_for_status(status=LlamaStackDistribution.Status.READY, timeout=600)
        yield lls_dist


@pytest.fixture(scope="class")
def llama_stack_distribution_deployment(
    admin_client: DynamicClient,
    llama_stack_distribution: LlamaStackDistribution,
) -> Generator[Deployment, Any, Any]:
    deployment = Deployment(
        client=admin_client,
        namespace=llama_stack_distribution.namespace,
        name=llama_stack_distribution.name,
    )

    deployment.wait(timeout=Timeout.TIMEOUT_2MIN)
    yield deployment


@pytest.fixture(scope="class")
def llama_stack_client(
    admin_client: DynamicClient,
    llama_stack_distribution_deployment: Deployment,
) -> Generator[LlamaStackClient, Any, Any]:
    """
    Returns a ready to use LlamaStackClient,  enabling port forwarding
    from the llama-stack-server service:8321 to localhost:8321

    Args:
        admin_client (DynamicClient): Kubernetes dynamic client for cluster operations
        llama_stack_distribution_deployment (Deployment): LlamaStack distribution deployment resource

    Yields:
        Generator[LlamaStackClient, Any, Any]: Configured LlamaStackClient for RAG testing
    """
    try:
        with portforward.forward(
            pod_or_service=f"{llama_stack_distribution_deployment.name}-service",
            namespace=llama_stack_distribution_deployment.namespace,
            from_port=8321,
            to_port=8321,
            waiting=15,
        ):
            client = LlamaStackClient(
                base_url="http://localhost:8321",
                timeout=120.0,
            )
            wait_for_llama_stack_client_ready(client=client)
            yield client
    except Exception as e:
        LOGGER.error(f"Failed to set up port forwarding: {e}")
        raise


@pytest.fixture(scope="class")
def llama_stack_models(llama_stack_client: LlamaStackClient) -> ModelInfo:
    """
    Returns model information from the LlamaStack client.

    Provides:
        - model_id: The identifier of the LLM model
        - embedding_model: The embedding model object
        - embedding_dimension: The dimension of the embedding model

    Args:
        llama_stack_client: The configured LlamaStackClient

    Returns:
        ModelInfo: NamedTuple containing model information
    """
    models = llama_stack_client.models.list()
    model_id = next(m for m in models if m.api_model_type == "llm").identifier

    embedding_model = next(m for m in models if m.api_model_type == "embedding")
    embedding_dimension = embedding_model.metadata["embedding_dimension"]

    return ModelInfo(model_id=model_id, embedding_model=embedding_model, embedding_dimension=embedding_dimension)


@pytest.fixture(scope="class")
def vector_store(llama_stack_client: LlamaStackClient, llama_stack_models: ModelInfo) -> Generator[Any, None, None]:
    """
    Creates a vector store for testing and automatically cleans it up.

    This fixture creates a vector store, yields it to the test,
    and ensures it's deleted after the test completes (whether it passes or fails).

    Args:
        llama_stack_client: The configured LlamaStackClient
        llama_stack_models: Model information including embedding model details

    Yields:
        Vector store object that can be used in tests
    """
    # Setup
    vector_store = llama_stack_client.vector_stores.create(
        name="test_vector_store",
        embedding_model=llama_stack_models.embedding_model.identifier,
        embedding_dimension=llama_stack_models.embedding_dimension,
    )

    yield vector_store

    try:
        llama_stack_client.vector_stores.delete(id=vector_store.id)
        LOGGER.info(f"Deleted vector store {vector_store.id}")
    except Exception as e:
        LOGGER.warning(f"Failed to delete vector store {vector_store.id}: {e}")


@retry(wait_timeout=Timeout.TIMEOUT_1MIN, sleep=5)
def _download_and_upload_file(url: str, llama_stack_client: LlamaStackClient, vector_store: Any) -> bool:
    """
    Downloads a file from URL and uploads it to the vector store.

    Args:
        url: The URL to download the file from
        llama_stack_client: The configured LlamaStackClient
        vector_store: The vector store to upload the file to

    Returns:
        bool: True if successful, raises exception if failed
    """
    import requests

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Save file locally first and pretend it's a txt file, not sure why this is needed
        # but it works locally without it,
        # though llama stack version is the newer one.
        file_name = url.split("/")[-1]
        local_file_name = file_name.replace(".rst", ".txt")
        with tempfile.NamedTemporaryFile(mode="wb", suffix=f"_{local_file_name}") as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

            # Upload saved file to LlamaStack
            with open(temp_file_path, "rb") as file_to_upload:
                uploaded_file = llama_stack_client.files.create(file=file_to_upload, purpose="assistants")

            # Add file to vector store
            llama_stack_client.vector_stores.files.create(vector_store_id=vector_store.id, file_id=uploaded_file.id)

        return True

    except (requests.exceptions.RequestException, Exception) as e:
        LOGGER.warning(f"Failed to download and upload file {url}: {e}")
        raise


@pytest.fixture(scope="class")
def vector_store_with_docs(llama_stack_client: LlamaStackClient, vector_store: Any) -> Generator[Any, None, None]:
    """
    Creates a vector store with TorchTune documentation files uploaded.

    This fixture depends on the vector_store fixture and uploads the TorchTune
    documentation files to the vector store for testing purposes. The files
    are automatically cleaned up after the test completes.

    Args:
        llama_stack_client: The configured LlamaStackClient
        vector_store: The vector store fixture to upload files to

    Yields:
        Vector store object with uploaded TorchTune documentation files
    """
    # Download TorchTune documentation files
    urls = [
        "llama3.rst",
        "chat.rst",
        "lora_finetune.rst",
        "qat_finetune.rst",
        "memory_optimizations.rst",
    ]

    base_url = "https://raw.githubusercontent.com/pytorch/torchtune/refs/tags/v0.6.1/docs/source/tutorials/"

    for file_name in urls:
        url = f"{base_url}{file_name}"
        _download_and_upload_file(url=url, llama_stack_client=llama_stack_client, vector_store=vector_store)

    yield vector_store
