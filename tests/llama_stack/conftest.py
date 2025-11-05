from typing import Generator, Any, Dict, Callable
import os
import portforward
import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from llama_stack_client import LlamaStackClient
from llama_stack_client.types.vector_store import VectorStore
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from ocp_resources.llama_stack_distribution import LlamaStackDistribution
from ocp_resources.namespace import Namespace
from semver import Version
from simple_logger.logger import get_logger
from utilities.general import generate_random_name
from tests.llama_stack.utils import (
    create_llama_stack_distribution,
    wait_for_llama_stack_client_ready,
    vector_store_create_file_from_url,
)
from utilities.constants import DscComponents
from utilities.data_science_cluster_utils import update_components_in_dsc
from tests.llama_stack.constants import (
    LLS_OPENSHIFT_MINIMAL_VERSION,
    ModelInfo,
)


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
    vector_io_provider_deployment_config_factory: Callable[[str], list[Dict[str, str]]],
) -> Dict[str, Any]:
    """
    Generate server configuration for LlamaStack distribution deployment and deploy vector I/O provider resources.

    This fixture creates a comprehensive server configuration dictionary that includes
    container specifications, environment variables, and optional storage settings.
    The configuration is built based on test parameters and environment variables.
    Additionally, it deploys the specified vector I/O provider (e.g., Milvus) and configures
    the necessary environment variables for the provider integration.

    Args:
        request: Pytest fixture request object containing test parameters
        vector_io_provider_deployment_config_factory: Factory function to deploy vector I/O providers
            and return their configuration environment variables

    Returns:
        Dict containing server configuration with the following structure:
        - containerSpec: Container resource limits, environment variables, and port
        - distribution: Distribution name (defaults to "rh-dev")
        - storage: Optional storage size configuration

    Environment Variables:
        The fixture configures the following environment variables:
        - INFERENCE_MODEL: Model identifier for inference
        - VLLM_API_TOKEN: API token for VLLM service
        - VLLM_URL: URL for VLLM service endpoint
        - VLLM_TLS_VERIFY: TLS verification setting (defaults to "false")
        - FMS_ORCHESTRATOR_URL: FMS orchestrator service URL
        - Vector I/O provider specific variables (deployed via factory):
          * For "milvus": MILVUS_DB_PATH
          * For "milvus-remote": MILVUS_ENDPOINT, MILVUS_TOKEN, MILVUS_CONSISTENCY_LEVEL

    Test Parameters:
        The fixture accepts the following optional parameters via request.param:
        - inference_model: Override for INFERENCE_MODEL environment variable
        - vllm_api_token: Override for VLLM_API_TOKEN environment variable
        - vllm_url_fixture: Fixture name to get VLLM URL from
        - fms_orchestrator_url_fixture: Fixture name to get FMS orchestrator URL from
        - vector_io_provider: Vector I/O provider type ("milvus" or "milvus-remote")
        - llama_stack_storage_size: Storage size for the deployment

    Example:
        @pytest.mark.parametrize("llama_stack_server_config",
                                [{"vector_io_provider": "milvus-remote"}],
                                indirect=True)
        def test_with_remote_milvus(llama_stack_server_config):
            # Test will use remote Milvus configuration
            pass
    """

    env_vars = []
    params = getattr(request, "param", {})

    # INFERENCE_MODEL
    if params.get("inference_model"):
        inference_model = str(params.get("inference_model"))
    else:
        inference_model = os.getenv("LLS_CORE_INFERENCE_MODEL", "")
    env_vars.append({"name": "INFERENCE_MODEL", "value": inference_model})

    # VLLM_API_TOKEN
    if params.get("vllm_api_token"):
        vllm_api_token = str(params.get("vllm_api_token"))
    else:
        vllm_api_token = os.getenv("LLS_CORE_VLLM_API_TOKEN", "")
    env_vars.append({"name": "VLLM_API_TOKEN", "value": vllm_api_token})

    # LLS_CORE_VLLM_URL
    if params.get("vllm_url_fixture"):
        vllm_url = str(request.getfixturevalue(argname=params.get("vllm_url_fixture")))
    else:
        vllm_url = os.getenv("LLS_CORE_VLLM_URL", "")
    env_vars.append({"name": "VLLM_URL", "value": vllm_url})

    # VLLM_TLS_VERIFY
    env_vars.append({"name": "VLLM_TLS_VERIFY", "value": "false"})

    # FMS_ORCHESTRATOR_URL
    if params.get("fms_orchestrator_url_fixture"):
        fms_orchestrator_url = str(request.getfixturevalue(argname=params.get("fms_orchestrator_url_fixture")))
    else:
        fms_orchestrator_url = "http://localhost"
    env_vars.append({"name": "FMS_ORCHESTRATOR_URL", "value": fms_orchestrator_url})

    # Depending on parameter vector_io_provider, deploy vector_io provider and obtain required env_vars
    vector_io_provider = params.get("vector_io_provider") or "milvus"
    env_vars_vector_io = vector_io_provider_deployment_config_factory(provider_name=vector_io_provider)
    env_vars.extend(env_vars_vector_io)

    server_config: Dict[str, Any] = {
        "containerSpec": {
            "resources": {
                "requests": {"cpu": "250m", "memory": "500Mi"},
                "limits": {"cpu": "2", "memory": "12Gi"},
            },
            "env": env_vars,
            "name": "llama-stack",
            "port": 8321,
        },
        "distribution": {"name": "rh-dev"},
    }

    if params.get("llama_stack_storage_size"):
        storage_size = params.get("llama_stack_storage_size")
        server_config["storage"] = {"size": storage_size}

    return server_config


@pytest.fixture(scope="class")
def unprivileged_llama_stack_distribution(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    enabled_llama_stack_operator: DataScienceCluster,
    llama_stack_server_config: Dict[str, Any],
) -> Generator[LlamaStackDistribution, None, None]:
    # Distribution name needs a random substring due to bug RHAIENG-999 / RHAIENG-1139
    distribution_name = generate_random_name(prefix="llama-stack-distribution")
    with create_llama_stack_distribution(
        client=unprivileged_client,
        name=distribution_name,
        namespace=unprivileged_model_namespace.name,
        replicas=1,
        server=llama_stack_server_config,
    ) as lls_dist:
        lls_dist.wait_for_status(status=LlamaStackDistribution.Status.READY, timeout=600)
        yield lls_dist


@pytest.fixture(scope="class")
def llama_stack_distribution(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    enabled_llama_stack_operator: DataScienceCluster,
    llama_stack_server_config: Dict[str, Any],
) -> Generator[LlamaStackDistribution, None, None]:
    # Distribution name needs a random substring due to bug RHAIENG-999 / RHAIENG-1139
    distribution_name = generate_random_name(prefix="llama-stack-distribution")
    with create_llama_stack_distribution(
        client=admin_client,
        name=distribution_name,
        namespace=model_namespace.name,
        replicas=1,
        server=llama_stack_server_config,
    ) as lls_dist:
        lls_dist.wait_for_status(status=LlamaStackDistribution.Status.READY, timeout=600)
        yield lls_dist


def _get_llama_stack_distribution_deployment(
    client: DynamicClient,
    llama_stack_distribution: LlamaStackDistribution,
) -> Generator[Deployment, Any, Any]:
    """
    Returns the Deployment resource for a given LlamaStackDistribution.
    Note: The deployment is created by the operator; this function retrieves it.

    Args:
        client (DynamicClient): Kubernetes client
        llama_stack_distribution (LlamaStackDistribution): LlamaStack distribution resource

    Yields:
        Generator[Deployment, Any, Any]: Deployment resource
    """
    deployment = Deployment(
        client=client,
        namespace=llama_stack_distribution.namespace,
        name=llama_stack_distribution.name,
        min_ready_seconds=10,
    )

    deployment.wait(timeout=120)
    yield deployment


@pytest.fixture(scope="session", autouse=True)
def skip_llama_stack_if_not_supported_openshift_version(
    admin_client: DynamicClient, openshift_version: Version
) -> None:
    """Skip llama-stack tests if OpenShift version is not supported (< 4.17) by llama-stack-operator"""
    if openshift_version < LLS_OPENSHIFT_MINIMAL_VERSION:
        message = (
            f"Skipping llama-stack tests, as llama-stack-operator is not supported "
            f"on OpenShift {openshift_version} ({LLS_OPENSHIFT_MINIMAL_VERSION} or newer required)"
        )
        LOGGER.info(message)
        pytest.skip(reason=message)


@pytest.fixture(scope="class")
def unprivileged_llama_stack_distribution_deployment(
    unprivileged_client: DynamicClient,
    unprivileged_llama_stack_distribution: LlamaStackDistribution,
) -> Generator[Deployment, Any, Any]:
    """
    Returns a deployment resource for unprivileged LlamaStack distribution.

    Args:
        unprivileged_client (DynamicClient): Unprivileged Kubernetes client
        unprivileged_llama_stack_distribution (LlamaStackDistribution): Unprivileged LlamaStack distribution resource

    Yields:
        Generator[Deployment, Any, Any]: Deployment resource
    """
    yield from _get_llama_stack_distribution_deployment(
        client=unprivileged_client, llama_stack_distribution=unprivileged_llama_stack_distribution
    )


@pytest.fixture(scope="class")
def llama_stack_distribution_deployment(
    admin_client: DynamicClient,
    llama_stack_distribution: LlamaStackDistribution,
) -> Generator[Deployment, Any, Any]:
    """
    Returns a deployment resource for admin LlamaStack distribution.

    Args:
        admin_client (DynamicClient): Admin Kubernetes client
        llama_stack_distribution (LlamaStackDistribution): LlamaStack distribution resource

    Yields:
        Generator[Deployment, Any, Any]: Deployment resource
    """
    yield from _get_llama_stack_distribution_deployment(
        client=admin_client, llama_stack_distribution=llama_stack_distribution
    )


def _create_llama_stack_client(
    llama_stack_distribution_deployment: Deployment,
) -> Generator[LlamaStackClient, Any, Any]:
    try:
        with portforward.forward(
            pod_or_service=f"{llama_stack_distribution_deployment.name}-service",
            namespace=llama_stack_distribution_deployment.namespace,
            from_port=8321,
            to_port=8321,
            waiting=30,
        ):
            client = LlamaStackClient(
                base_url="http://localhost:8321",
                timeout=180.0,
            )
            wait_for_llama_stack_client_ready(client=client)
            yield client
    except Exception as e:
        LOGGER.error(f"Failed to set up port forwarding: {e}")
        raise


@pytest.fixture(scope="class")
def unprivileged_llama_stack_client(
    unprivileged_llama_stack_distribution_deployment: Deployment,
) -> Generator[LlamaStackClient, Any, Any]:
    """
    Returns a ready to use LlamaStackClient for unprivileged deployment.

    Args:
        unprivileged_llama_stack_distribution_deployment (Deployment): LlamaStack distribution deployment resource

    Yields:
        Generator[LlamaStackClient, Any, Any]: Configured LlamaStackClient for RAG testing
    """
    yield from _create_llama_stack_client(
        llama_stack_distribution_deployment=unprivileged_llama_stack_distribution_deployment
    )


@pytest.fixture(scope="class")
def llama_stack_client(
    llama_stack_distribution_deployment: Deployment,
) -> Generator[LlamaStackClient, Any, Any]:
    """
    Returns a ready to use LlamaStackClient.

    Args:
        llama_stack_distribution_deployment (Deployment): LlamaStack distribution deployment resource

    Yields:
        Generator[LlamaStackClient, Any, Any]: Configured LlamaStackClient for RAG testing
    """
    yield from _create_llama_stack_client(llama_stack_distribution_deployment=llama_stack_distribution_deployment)


@pytest.fixture(scope="class")
def llama_stack_models(unprivileged_llama_stack_client: LlamaStackClient) -> ModelInfo:
    """
    Returns model information from the LlamaStack client.

    Provides:
        - model_id: The identifier of the LLM model
        - embedding_model: The embedding model object
        - embedding_dimension: The dimension of the embedding model

    Args:
        unprivileged_llama_stack_client: The configured LlamaStackClient

    Returns:
        ModelInfo: NamedTuple containing model information
    """
    models = unprivileged_llama_stack_client.models.list()
    model_id = next(m for m in models if m.api_model_type == "llm").identifier

    embedding_model = next(m for m in models if m.api_model_type == "embedding")
    embedding_dimension = embedding_model.metadata["embedding_dimension"]

    return ModelInfo(model_id=model_id, embedding_model=embedding_model, embedding_dimension=embedding_dimension)


@pytest.fixture(scope="class")
def vector_store(
    unprivileged_llama_stack_client: LlamaStackClient,
    llama_stack_models: ModelInfo,
    request: FixtureRequest,
) -> Generator[VectorStore, None, None]:
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

    params = getattr(request, "param", {"vector_io_provider": "milvus"})
    vector_io_provider = str(params.get("vector_io_provider"))

    vector_store = unprivileged_llama_stack_client.vector_stores.create(
        name="test_vector_store",
        extra_body={
            "embedding_model": llama_stack_models.embedding_model.identifier,
            "embedding_dimension": llama_stack_models.embedding_dimension,
            "provider_id": vector_io_provider,
        },
    )
    LOGGER.info(f"vector_store successfully created (provider_id={vector_io_provider}, id={vector_store.id})")

    yield vector_store

    try:
        unprivileged_llama_stack_client.vector_stores.delete(vector_store_id=vector_store.id)
        LOGGER.info(f"Deleted vector store {vector_store.id}")
    except Exception as e:
        LOGGER.warning(f"Failed to delete vector store {vector_store.id}: {e}")


@pytest.fixture(scope="class")
def vector_store_with_example_docs(
    unprivileged_llama_stack_client: LlamaStackClient, vector_store: VectorStore
) -> Generator[VectorStore, None, None]:
    """
    Creates a vector store with TorchTune documentation files uploaded.

    This fixture depends on the vector_store fixture and uploads the TorchTune
    documentation files to the vector store for testing purposes. The files
    are automatically cleaned up after the test completes.

    Args:
        unprivileged_llama_stack_client: The configured LlamaStackClient
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
        vector_store_create_file_from_url(
            url=url, llama_stack_client=unprivileged_llama_stack_client, vector_store=vector_store
        )

    yield vector_store
