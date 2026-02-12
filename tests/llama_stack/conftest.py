from typing import Generator, Any, Dict, Callable
import os
import httpx
from ocp_resources.route import Route
from ocp_resources.resource import ResourceEditor
import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from llama_stack_client import LlamaStackClient
from llama_stack_client.types.vector_store import VectorStore
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment

from utilities.resources.llama_stack_distribution import LlamaStackDistribution
from ocp_resources.namespace import Namespace
from semver import Version
from simple_logger.logger import get_logger
from utilities.general import generate_random_name
from tests.llama_stack.utils import (
    create_llama_stack_distribution,
    wait_for_llama_stack_client_ready,
    vector_store_create_file_from_url,
    wait_for_unique_llama_stack_pod,
)
from utilities.constants import DscComponents, Annotations
from utilities.data_science_cluster_utils import update_components_in_dsc
from tests.llama_stack.constants import (
    LLS_OPENSHIFT_MINIMAL_VERSION,
    ModelInfo,
)
from ocp_resources.service import Service
from ocp_resources.secret import Secret

LOGGER = get_logger(name=__name__)

pytestmark = pytest.mark.skip_on_disconnected

POSTGRES_IMAGE = os.getenv(
    "LLS_VECTOR_IO_POSTGRES_IMAGE",
    (
        "registry.redhat.io/rhel9/postgresql-15@sha256:"
        "90ec347a35ab8a5d530c8d09f5347b13cc71df04f3b994bfa8b1a409b1171d59"  # postgres 15 # pragma: allowlist secret
    ),
)

POSTGRESQL_USER = os.getenv("LLS_VECTOR_IO_POSTGRESQL_USER", "ps_user")
POSTGRESQL_PASSWORD = os.getenv("LLS_VECTOR_IO_POSTGRESQL_PASSWORD", "ps_password")

LLAMA_STACK_DISTRIBUTION_SECRET_DATA = {
    "postgres-user": POSTGRESQL_USER,
    "postgres-password": POSTGRESQL_PASSWORD,
}

LLS_CORE_INFERENCE_MODEL = os.getenv("LLS_CORE_INFERENCE_MODEL", "")
LLS_CORE_VLLM_URL = os.getenv("LLS_CORE_VLLM_URL", "")
LLS_CORE_VLLM_API_TOKEN = os.getenv("LLS_CORE_VLLM_API_TOKEN", "")
LLS_CORE_VLLM_MAX_TOKENS = os.getenv("LLS_CORE_VLLM_MAX_TOKENS", "16384")
LLS_CORE_VLLM_TLS_VERIFY = os.getenv("LLS_CORE_VLLM_TLS_VERIFY", "true")

LLS_CORE_EMBEDDING_MODEL = os.getenv("LLS_CORE_EMBEDDING_MODEL", "nomic-embed-text-v1-5")
LLS_CORE_EMBEDDING_PROVIDER_MODEL_ID = os.getenv("LLS_CORE_EMBEDDING_PROVIDER_MODEL_ID", "nomic-embed-text-v1-5")
LLS_CORE_VLLM_EMBEDDING_URL = os.getenv(
    "LLS_CORE_VLLM_EMBEDDING_URL", "https://nomic-embed-text-v1-5.example.com:443/v1"
)
LLS_CORE_VLLM_EMBEDDING_API_TOKEN = os.getenv("LLS_CORE_VLLM_EMBEDDING_API_TOKEN", "fake")
LLS_CORE_VLLM_EMBEDDING_MAX_TOKENS = os.getenv("LLS_CORE_VLLM_EMBEDDING_MAX_TOKENS", "8192")
LLS_CORE_VLLM_EMBEDDING_TLS_VERIFY = os.getenv("LLS_CORE_VLLM_EMBEDDING_TLS_VERIFY", "true")

distribution_name = generate_random_name(prefix="llama-stack-distribution")


def _cleanup_s3_files(
    bucket_name: str,
    endpoint_url: str,
    region: str,
    access_key_id: str,
    secret_access_key: str,
) -> None:
    """
    Clean up files from S3 bucket that were uploaded during tests.

    Args:
        bucket_name: S3 bucket name
        endpoint_url: S3 endpoint URL
        region: S3 region
        access_key_id: AWS access key ID
        secret_access_key: AWS secret access key
    """

    try:
        import boto3
        from botocore.exceptions import ClientError

        s3_client = boto3.client(
            service_name="s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region,
        )

        response = s3_client.list_objects_v2(Bucket=bucket_name)

        if "Contents" not in response:
            LOGGER.info("No files found to clean up from S3")
            return

        # We only want to delete files that start with "file-"
        for obj in response["Contents"]:
            key = obj["Key"]
            if key.startswith("file-"):
                s3_client.delete_object(Bucket=bucket_name, Key=key)
                LOGGER.debug(f"Deleted file from S3: {key}")

        response = s3_client.list_objects_v2(Bucket=bucket_name)

        if "Contents" not in response:
            LOGGER.info("No files found to clean up from S3")
            return

    except ClientError as e:
        LOGGER.warning(f"Failed to clean up S3 files: {e}")


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
    files_provider_config_factory: Callable[[str], list[Dict[str, str]]],
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
        files_provider_config_factory: Factory function to configure files storage providers
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
        - ENABLE_SENTENCE_TRANSFORMERS: Enable sentence-transformers embeddings (set to "true")
        - EMBEDDING_PROVIDER: Embeddings provider to use (set to "sentence-transformers")
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
        - embedding_model: Embedding model identifier for inference
        - kubeflow_llama_stack_url: LlamaStack service URL for Kubeflow
        - kubeflow_pipelines_endpoint: Kubeflow Pipelines API endpoint URL
        - kubeflow_namespace: Namespace for Kubeflow resources
        - kubeflow_base_image: Base container image for Kubeflow pipelines
        - kubeflow_results_s3_prefix: S3 prefix for storing Kubeflow results
        - kubeflow_s3_credentials_secret_name: Secret name for S3 credentials
        - kubeflow_pipelines_token: Authentication token for Kubeflow Pipelines

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
        inference_model = LLS_CORE_INFERENCE_MODEL
    env_vars.append({"name": "INFERENCE_MODEL", "value": inference_model})

    if params.get("vllm_api_token"):
        vllm_api_token = str(params.get("vllm_api_token"))
    else:
        vllm_api_token = LLS_CORE_VLLM_API_TOKEN
    env_vars.append({"name": "VLLM_API_TOKEN", "value": vllm_api_token})

    if params.get("vllm_url_fixture"):
        vllm_url = str(request.getfixturevalue(argname=params.get("vllm_url_fixture")))
    else:
        vllm_url = LLS_CORE_VLLM_URL
    env_vars.append({"name": "VLLM_URL", "value": vllm_url})

    env_vars.append({"name": "VLLM_TLS_VERIFY", "value": LLS_CORE_VLLM_TLS_VERIFY})
    env_vars.append({"name": "VLLM_MAX_TOKENS", "value": LLS_CORE_VLLM_MAX_TOKENS})

    # FMS_ORCHESTRATOR_URL
    if params.get("fms_orchestrator_url_fixture"):
        fms_orchestrator_url = str(request.getfixturevalue(argname=params.get("fms_orchestrator_url_fixture")))
    else:
        fms_orchestrator_url = "http://localhost"
    env_vars.append({"name": "FMS_ORCHESTRATOR_URL", "value": fms_orchestrator_url})

    # EMBEDDING_MODEL
    embedding_provider = params.get("embedding_provider") or "vllm-embedding"

    if embedding_provider == "vllm-embedding":
        env_vars.append({"name": "EMBEDDING_MODEL", "value": LLS_CORE_EMBEDDING_MODEL})
        env_vars.append({"name": "EMBEDDING_PROVIDER_MODEL_ID", "value": LLS_CORE_EMBEDDING_PROVIDER_MODEL_ID})
        env_vars.append({"name": "VLLM_EMBEDDING_URL", "value": LLS_CORE_VLLM_EMBEDDING_URL})
        env_vars.append({"name": "VLLM_EMBEDDING_API_TOKEN", "value": LLS_CORE_VLLM_EMBEDDING_API_TOKEN})
        env_vars.append({"name": "VLLM_EMBEDDING_MAX_TOKENS", "value": LLS_CORE_VLLM_EMBEDDING_MAX_TOKENS})
        env_vars.append({"name": "VLLM_EMBEDDING_TLS_VERIFY", "value": LLS_CORE_VLLM_EMBEDDING_TLS_VERIFY})
    elif embedding_provider == "sentence-transformers":
        env_vars.append({"name": "ENABLE_SENTENCE_TRANSFORMERS", "value": "true"})
        env_vars.append({"name": "EMBEDDING_PROVIDER", "value": "sentence-transformers"})
    else:
        raise ValueError(f"Unsupported embeddings provider: {embedding_provider}")

    # TRUSTYAI_EMBEDDING_MODEL
    trustyai_embedding_model = params.get("trustyai_embedding_model")
    if trustyai_embedding_model:
        env_vars.append({"name": "TRUSTYAI_EMBEDDING_MODEL", "value": trustyai_embedding_model})

    # Kubeflow-related environment variables
    if params.get("enable_ragas_remote"):
        # Get fixtures only when Ragas Remote/Kubeflow is enabled
        model_namespace = request.getfixturevalue(argname="model_namespace")
        current_client_token = request.getfixturevalue(argname="current_client_token")
        dspa_route = request.getfixturevalue(argname="dspa_route")
        dspa_s3_secret = request.getfixturevalue(argname="dspa_s3_secret")

        # KUBEFLOW_LLAMA_STACK_URL: Build from LlamaStackDistribution service
        env_vars.append({
            "name": "KUBEFLOW_LLAMA_STACK_URL",
            "value": f"http://{distribution_name}-service.{model_namespace.name}.svc.cluster.local:8321",
        })

        # KUBEFLOW_PIPELINES_ENDPOINT: Get from DSPA route
        env_vars.append({"name": "KUBEFLOW_PIPELINES_ENDPOINT", "value": f"https://{dspa_route.instance.spec.host}"})

        # KUBEFLOW_NAMESPACE: Use model namespace
        env_vars.append({"name": "KUBEFLOW_NAMESPACE", "value": model_namespace.name})

        # KUBEFLOW_BASE_IMAGE
        env_vars.append({
            "name": "KUBEFLOW_BASE_IMAGE",
            "value": params.get(
                "kubeflow_base_image",
                "quay.io/diegosquayorg/my-ragas-provider-image"
                "@sha256:3749096c47f7536d6be2a7932e691abebacd578bafbe65bad2f7db475e2b93fb",
            ),
        })

        # KUBEFLOW_RESULTS_S3_PREFIX: Build from MinIO bucket
        env_vars.append({
            "name": "KUBEFLOW_RESULTS_S3_PREFIX",
            "value": params.get("kubeflow_results_s3_prefix", "s3://llms/ragas-results"),
        })

        # KUBEFLOW_S3_CREDENTIALS_SECRET_NAME: Use DSPA secret name
        env_vars.append({"name": "KUBEFLOW_S3_CREDENTIALS_SECRET_NAME", "value": dspa_s3_secret.name})

        # KUBEFLOW_PIPELINES_TOKEN: Get from current client token
        env_vars.append({"name": "KUBEFLOW_PIPELINES_TOKEN", "value": str(current_client_token)})

    # POSTGRESQL environment variables for sql_default and kvstore_default
    env_vars.append({"name": "POSTGRES_HOST", "value": "vector-io-postgres-service"})
    env_vars.append({"name": "POSTGRES_PORT", "value": "5432"})
    env_vars.append(
        {
            "name": "POSTGRES_USER",
            "valueFrom": {"secretKeyRef": {"name": "llamastack-distribution-secret", "key": "postgres-user"}},
        },
    )
    env_vars.append(
        {
            "name": "POSTGRES_PASSWORD",
            "valueFrom": {"secretKeyRef": {"name": "llamastack-distribution-secret", "key": "postgres-password"}},
        },
    )
    env_vars.append({"name": "POSTGRES_DB", "value": "ps_db"})
    env_vars.append({"name": "POSTGRES_TABLE_NAME", "value": "llamastack_kvstore"})

    # Depending on parameter files_provider, configure files provider and obtain required env_vars
    files_provider = params.get("files_provider") or "local"
    env_vars_files = files_provider_config_factory(provider_name=files_provider)
    env_vars.extend(env_vars_files)

    # Depending on parameter vector_io_provider, deploy vector_io provider and obtain required env_vars
    vector_io_provider = params.get("vector_io_provider") or "milvus"
    env_vars_vector_io = vector_io_provider_deployment_config_factory(provider_name=vector_io_provider)
    env_vars.extend(env_vars_vector_io)

    server_config: Dict[str, Any] = {
        "containerSpec": {
            "resources": {
                "requests": {"cpu": "1", "memory": "3Gi"},
                "limits": {"cpu": "3", "memory": "6Gi"},
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
def llama_stack_distribution_secret(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[Secret, Any, Any]:
    with Secret(
        client=admin_client,
        namespace=model_namespace.name,
        name="llamastack-distribution-secret",
        type="Opaque",
        string_data=LLAMA_STACK_DISTRIBUTION_SECRET_DATA,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def unprivileged_llama_stack_distribution_secret(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[Secret, Any, Any]:
    with Secret(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        name="llamastack-distribution-secret",
        type="Opaque",
        string_data=LLAMA_STACK_DISTRIBUTION_SECRET_DATA,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def unprivileged_llama_stack_distribution(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    enabled_llama_stack_operator: DataScienceCluster,
    request: FixtureRequest,
    llama_stack_server_config: Dict[str, Any],
    ci_s3_bucket_name: str,
    ci_s3_bucket_endpoint: str,
    ci_s3_bucket_region: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    unprivileged_llama_stack_distribution_secret: Secret,
    unprivileged_postgres_deployment: Deployment,
    unprivileged_postgres_service: Service,
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

        try:
            env_vars = llama_stack_server_config.get("containerSpec", {}).get("env", [])
            enable_s3 = any(env.get("name") == "ENABLE_S3" and env.get("value") == "s3" for env in env_vars)

            if enable_s3:
                try:
                    _cleanup_s3_files(
                        bucket_name=ci_s3_bucket_name,
                        endpoint_url=ci_s3_bucket_endpoint,
                        region=ci_s3_bucket_region,
                        access_key_id=aws_access_key_id,
                        secret_access_key=aws_secret_access_key,
                    )
                except Exception as e:
                    LOGGER.warning(f"Failed to clean up S3 files: {e}")

        except Exception as e:
            LOGGER.warning(f"Failed to clean up S3 files: {e}")


@pytest.fixture(scope="class")
def llama_stack_distribution(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    enabled_llama_stack_operator: DataScienceCluster,
    request: FixtureRequest,
    llama_stack_server_config: Dict[str, Any],
    ci_s3_bucket_name: str,
    ci_s3_bucket_endpoint: str,
    ci_s3_bucket_region: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    llama_stack_distribution_secret: Secret,
    postgres_deployment: Deployment,
    postgres_service: Service,
) -> Generator[LlamaStackDistribution, None, None]:
    # Distribution name needs a random substring due to bug RHAIENG-999 / RHAIENG-1139
    with create_llama_stack_distribution(
        client=admin_client,
        name=distribution_name,
        namespace=model_namespace.name,
        replicas=1,
        server=llama_stack_server_config,
    ) as lls_dist:
        lls_dist.wait_for_status(status=LlamaStackDistribution.Status.READY, timeout=600)
        yield lls_dist

        try:
            env_vars = llama_stack_server_config.get("containerSpec", {}).get("env", [])
            enable_s3 = any(env.get("name") == "ENABLE_S3" and env.get("value") == "s3" for env in env_vars)

            if enable_s3:
                try:
                    _cleanup_s3_files(
                        bucket_name=ci_s3_bucket_name,
                        endpoint_url=ci_s3_bucket_endpoint,
                        region=ci_s3_bucket_region,
                        access_key_id=aws_access_key_id,
                        secret_access_key=aws_secret_access_key,
                    )
                except Exception as e:
                    LOGGER.warning(f"Failed to clean up S3 files: {e}")

        except Exception as e:
            LOGGER.warning(f"Failed to clean up S3 files: {e}")


def _get_llama_stack_distribution_deployment(
    client: DynamicClient,
    llama_stack_distribution: LlamaStackDistribution,
) -> Generator[Deployment, Any, Any]:
    """
    Returns the Deployment resource for a given LlamaStackDistribution.
    Note: The deployment is created by the operator; this function retrieves it.
    Includes a workaround for RHAIENG-1819 to ensure exactly one pod exists.

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
    deployment.timeout_seconds = 120
    deployment.wait(timeout=120)
    deployment.wait_for_replicas()
    # Workaround for RHAIENG-1819 (Incorrect number of llama-stack pods deployed after
    # creating LlamaStackDistribution after setting custom ca bundle in DSCI)
    wait_for_unique_llama_stack_pod(client=client, namespace=llama_stack_distribution.namespace)
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


def _create_llama_stack_test_route(
    client: DynamicClient,
    namespace: Namespace,
    deployment: Deployment,
) -> Generator[Route, Any, Any]:
    """
    Creates a Route for LlamaStack distribution with TLS configuration.

    Args:
        client: Kubernetes client
        namespace: Namespace where the route will be created
        deployment: Deployment resource to create the route for

    Yields:
        Generator[Route, Any, Any]: Route resource with TLS edge termination
    """
    route_name = generate_random_name(prefix="llama-stack", length=12)
    with Route(
        client=client,
        namespace=namespace.name,
        name=route_name,
        service=f"{deployment.name}-service",
        wait_for_resource=True,
    ) as route:
        with ResourceEditor(
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
        ):
            route.wait(timeout=60)
            yield route


@pytest.fixture(scope="class")
def unprivileged_llama_stack_test_route(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    unprivileged_llama_stack_distribution_deployment: Deployment,
) -> Generator[Route, Any, Any]:
    yield from _create_llama_stack_test_route(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace,
        deployment=unprivileged_llama_stack_distribution_deployment,
    )


@pytest.fixture(scope="class")
def llama_stack_test_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    llama_stack_distribution_deployment: Deployment,
) -> Generator[Route, Any, Any]:
    yield from _create_llama_stack_test_route(
        client=admin_client,
        namespace=model_namespace,
        deployment=llama_stack_distribution_deployment,
    )


def _create_llama_stack_client(
    route: Route,
) -> Generator[LlamaStackClient, Any, Any]:
    # LLS_CLIENT_VERIFY_SSL is false by default to be able to test with Self-Signed certificates
    verifySSL = os.getenv("LLS_CLIENT_VERIFY_SSL", "false").lower() == "true"
    http_client = httpx.Client(verify=verifySSL, timeout=240)
    try:
        client = LlamaStackClient(
            base_url=f"https://{route.host}",
            max_retries=3,
            http_client=http_client,
        )
        wait_for_llama_stack_client_ready(client=client)
        yield client
    finally:
        http_client.close()


@pytest.fixture(scope="class")
def unprivileged_llama_stack_client(
    unprivileged_llama_stack_test_route: Route,
) -> Generator[LlamaStackClient, Any, Any]:
    """
    Returns a ready to use LlamaStackClient for unprivileged deployment.

    Args:
        unprivileged_llama_stack_test_route (Route): Route resource for unprivileged LlamaStack distribution

    Yields:
        Generator[LlamaStackClient, Any, Any]: Configured LlamaStackClient for RAG testing
    """
    yield from _create_llama_stack_client(
        route=unprivileged_llama_stack_test_route,
    )


@pytest.fixture(scope="class")
def llama_stack_client(
    llama_stack_test_route: Route,
) -> Generator[LlamaStackClient, Any, Any]:
    """
    Returns a ready to use LlamaStackClient.

    Args:
        llama_stack_test_route (Route): Route resource for LlamaStack distribution

    Yields:
        Generator[LlamaStackClient, Any, Any]: Configured LlamaStackClient for RAG testing
    """
    yield from _create_llama_stack_client(
        route=llama_stack_test_route,
    )


@pytest.fixture(scope="class")
def llama_stack_models(unprivileged_llama_stack_client: LlamaStackClient) -> ModelInfo:
    """
    Returns model information from the LlamaStack client.

    Selects the embedding model based on available providers with the following priority:
    1. sentence-transformers provider (if present)
    2. vllm-embedding provider (if present)

    Provides:
        - model_id: The identifier of the LLM model
        - embedding_model: The embedding model object from the selected provider
        - embedding_dimension: The dimension of the embedding model

    Args:
        unprivileged_llama_stack_client: The configured LlamaStackClient

    Returns:
        ModelInfo: NamedTuple containing model information

    Raises:
        ValueError: If no embedding provider (sentence-transformers or vllm-embedding) is found

    """
    models = unprivileged_llama_stack_client.models.list()

    model_id = next(m for m in models if m.custom_metadata["model_type"] == "llm").id

    # Ensure getting the right embedding model depending on the available providers
    providers = unprivileged_llama_stack_client.providers.list()
    provider_ids = [p.provider_id for p in providers]
    if "sentence-transformers" in provider_ids:
        target_provider_id = "sentence-transformers"
    elif "vllm-embedding" in provider_ids:
        target_provider_id = "vllm-embedding"
    else:
        raise ValueError("No embedding provider found")

    embedding_model = next(
        m
        for m in models
        if m.custom_metadata["model_type"] == "embedding" and m.custom_metadata["provider_id"] == target_provider_id
    )
    embedding_dimension = int(embedding_model.custom_metadata["embedding_dimension"])

    LOGGER.info(f"Detected model: {model_id}")
    LOGGER.info(f"Detected embedding_model: {embedding_model.id}")
    LOGGER.info(f"Detected embedding_dimension: {embedding_dimension}")

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
            "embedding_model": llama_stack_models.embedding_model.id,
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


@pytest.fixture(scope="class")
def unprivileged_postgres_service(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    unprivileged_postgres_deployment: Deployment,
) -> Generator[Service, Any, Any]:
    """Create a service for the unprivileged postgres deployment."""
    with Service(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        name="vector-io-postgres-service",
        ports=[
            {
                "port": 5432,
                "targetPort": 5432,
            }
        ],
        selector={"app": "postgres"},
        wait_for_resource=True,
    ) as service:
        yield service


@pytest.fixture(scope="class")
def unprivileged_postgres_deployment(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[Deployment, Any, Any]:
    """Deploy a Postgres instance for vector I/O provider testing with unprivileged client."""
    with Deployment(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        name="vector-io-postgres-deployment",
        min_ready_seconds=5,
        replicas=1,
        selector={"matchLabels": {"app": "postgres"}},
        strategy={"type": "Recreate"},
        template=get_postgres_deployment_template(),
        teardown=True,
    ) as deployment:
        deployment.wait_for_replicas(deployed=True, timeout=240)
        yield deployment


@pytest.fixture(scope="class")
def postgres_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    postgres_deployment: Deployment,
) -> Generator[Service, Any, Any]:
    """Create a service for the postgres deployment."""
    with Service(
        client=admin_client,
        namespace=model_namespace.name,
        name="vector-io-postgres-service",
        ports=[
            {
                "port": 5432,
                "targetPort": 5432,
            }
        ],
        selector={"app": "postgres"},
        wait_for_resource=True,
    ) as service:
        yield service


@pytest.fixture(scope="class")
def postgres_deployment(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[Deployment, Any, Any]:
    """Deploy a Postgres instance for vector I/O provider testing."""
    with Deployment(
        client=admin_client,
        namespace=model_namespace.name,
        name="vector-io-postgres-deployment",
        min_ready_seconds=5,
        replicas=1,
        selector={"matchLabels": {"app": "postgres"}},
        strategy={"type": "Recreate"},
        template=get_postgres_deployment_template(),
        teardown=True,
    ) as deployment:
        deployment.wait_for_replicas(deployed=True, timeout=240)
        yield deployment


def get_postgres_deployment_template() -> Dict[str, Any]:
    """Return a Kubernetes deployment for PostgreSQL"""
    return {
        "metadata": {"labels": {"app": "postgres"}},
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
                                "secretKeyRef": {"name": "llamastack-distribution-secret", "key": "postgres-user"}
                            },
                        },
                        {
                            "name": "POSTGRESQL_PASSWORD",
                            "valueFrom": {
                                "secretKeyRef": {"name": "llamastack-distribution-secret", "key": "postgres-password"}
                            },
                        },
                    ],
                    "volumeMounts": [{"name": "postgresdata", "mountPath": "/var/lib/pgsql/data"}],
                },
            ],
            "volumes": [{"name": "postgresdata", "emptyDir": {}}],
        },
    }
