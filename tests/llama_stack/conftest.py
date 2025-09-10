import os
from typing import Generator, Any, Dict

import portforward
import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from llama_stack_client import LlamaStackClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from ocp_resources.service import Service
from ocp_resources.config_map import ConfigMap
from .utils import get_etcd_deployment_template, get_milvus_deployment_template
from ocp_resources.llama_stack_distribution import LlamaStackDistribution
from ocp_resources.namespace import Namespace
from simple_logger.logger import get_logger

from tests.llama_stack.utils import create_llama_stack_distribution, wait_for_llama_stack_client_ready
from utilities.constants import DscComponents, Timeout
from utilities.data_science_cluster_utils import update_components_in_dsc


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
            "command": ["/bin/sh", "-c", "llama stack run /etc/llama-stack/run.yaml"],  # Necessary for v2.24
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
                {"name": "MILVUS_ENDPOINT", "value": "http://rag-milvus-service:19530"},
                {"name": "MILVUS_TOKEN", "value": "root:Milvus"},
                {"name": "FMS_ORCHESTRATOR_URL", "value": fms_orchestrator_url},
            ],
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
def llama_stack_distribution(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    enabled_llama_stack_operator: DataScienceCluster,
    llama_stack_config_map: ConfigMap,
    remote_milvus_deployment: Deployment,
    milvus_service: Service,
    etcd_deployment: Deployment,
    etcd_service: Service,
    llama_stack_server_config: Dict[str, Any],
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
def etcd_deployment(
    model_namespace: Namespace,
    admin_client: DynamicClient,
) -> Generator[Deployment, Any, Any]:
    with Deployment(
        client=admin_client,
        namespace=model_namespace.name,
        name="rag-etcd-deployment",
        replicas=1,
        selector={"matchLabels": {"app": "etcd"}},
        strategy={"type": "Recreate"},
        template=get_etcd_deployment_template(),
        teardown=True,
    ) as deployment:
        deployment.wait_for_replicas(deployed=True, timeout=Timeout.TIMEOUT_2MIN)
        yield deployment


@pytest.fixture(scope="class")
def etcd_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[Service, Any, Any]:
    with Service(
        client=admin_client,
        namespace=model_namespace.name,
        name="rag-etcd-service",
        ports=[
            {
                "port": 2379,
                "targetPort": 2379,
            }
        ],
        selector={"app": "etcd"},
    ) as service:
        yield service


@pytest.fixture(scope="class")
def remote_milvus_deployment(
    model_namespace: Namespace,
    admin_client: DynamicClient,
    etcd_deployment: Deployment,
    etcd_service: Service,
) -> Generator[Deployment, Any, Any]:
    with Deployment(
        client=admin_client,
        namespace=model_namespace.name,
        name="rag-milvus-deployment",
        replicas=1,
        selector={"matchLabels": {"app": "milvus-standalone"}},
        strategy={"type": "Recreate"},
        template=get_milvus_deployment_template(),
        teardown=True,
    ) as deployment:
        deployment.wait_for_replicas(deployed=True, timeout=Timeout.TIMEOUT_2MIN)
        yield deployment


@pytest.fixture(scope="class")
def milvus_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[Service, Any, Any]:
    with Service(
        client=admin_client,
        namespace=model_namespace.name,
        name="rag-milvus-service",
        ports=[
            {
                "name": "grpc",
                "port": 19530,
                "targetPort": 19530,
            },
        ],
        selector={"app": "milvus-standalone"},
    ) as service:
        yield service


@pytest.fixture(scope="class")
def llama_stack_config_map(
    model_namespace: Namespace,
    admin_client: DynamicClient,
) -> Generator[ConfigMap, Any, Any]:
    with ConfigMap(
        client=admin_client,
        namespace=model_namespace.name,
        name="rag-llama-stack-config-map",
        data={
            "run.yaml": """# Llama Stack Configuration
version: "2"
image_name: rh
apis:
  - agents
  - datasetio
  - eval
  - inference
  - safety
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
    - provider_id: remote-milvus
      provider_type: remote::milvus
      config:
        uri: ${env.MILVUS_ENDPOINT:=http://localhost:19530}
        token: ${env.MILVUS_TOKEN:=root:Milvus}
        kvstore:
          type: sqlite
          db_path: ~/.llama/distributions/rh/milvus_remote_registry.db
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
external_providers_dir: /opt/app-root/src/.llama/providers.d
"""
        },
    ) as config_map:
        yield config_map
