import os
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import httpx
import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from llama_stack_client import APIError, LlamaStackClient
from llama_stack_client.types.vector_store import VectorStore
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from semver import Version
from simple_logger.logger import get_logger

from tests.llama_stack.constants import (
    LLAMA_STACK_DISTRIBUTION_SECRET_DATA,
    LLS_CORE_EMBEDDING_MODEL,
    LLS_CORE_EMBEDDING_PROVIDER_MODEL_ID,
    LLS_CORE_INFERENCE_MODEL,
    LLS_CORE_VLLM_EMBEDDING_MAX_TOKENS,
    LLS_CORE_VLLM_EMBEDDING_TLS_VERIFY,
    LLS_CORE_VLLM_EMBEDDING_URL,
    LLS_CORE_VLLM_MAX_TOKENS,
    LLS_CORE_VLLM_TLS_VERIFY,
    LLS_CORE_VLLM_URL,
    LLS_OPENSHIFT_MINIMAL_VERSION,
    POSTGRES_IMAGE,
    UPGRADE_DISTRIBUTION_NAME,
    ModelInfo,
)
from tests.llama_stack.utils import (
    create_llama_stack_distribution,
    vector_store_create_file_from_path,
    vector_store_create_file_from_url,
    wait_for_llama_stack_client_ready,
    wait_for_unique_llama_stack_pod,
)
from utilities.constants import Annotations, DscComponents
from utilities.data_science_cluster_utils import update_components_in_dsc
from utilities.general import generate_random_name
from utilities.resources.llama_stack_distribution import LlamaStackDistribution

LOGGER = get_logger(name=__name__)

pytestmark = pytest.mark.skip_on_disconnected


@pytest.fixture(scope="class")
def distribution_name(pytestconfig: pytest.Config) -> str:
    if pytestconfig.option.pre_upgrade or pytestconfig.option.post_upgrade:
        return UPGRADE_DISTRIBUTION_NAME
    return generate_random_name(prefix="llama-stack-distribution")


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
    pytestconfig: pytest.Config,
    distribution_name: str,
    vector_io_provider_deployment_config_factory: Callable[[str], list[dict[str, str]]],
    files_provider_config_factory: Callable[[str], list[dict[str, str]]],
) -> dict[str, Any]:
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

    env_vars.append(
        {
            "name": "VLLM_API_TOKEN",
            "valueFrom": {"secretKeyRef": {"name": "llamastack-distribution-secret", "key": "vllm-api-token"}},
        },
    )

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
        env_vars.append(
            {
                "name": "VLLM_EMBEDDING_API_TOKEN",
                "valueFrom": {
                    "secretKeyRef": {"name": "llamastack-distribution-secret", "key": "vllm-embedding-api-token"}
                },
            },
        )
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

    server_config: dict[str, Any] = {
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
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[Secret, Any, Any]:
    secret = Secret(
        client=admin_client,
        namespace=model_namespace.name,
        name="llamastack-distribution-secret",
        type="Opaque",
        string_data=LLAMA_STACK_DISTRIBUTION_SECRET_DATA,
        ensure_exists=pytestconfig.option.post_upgrade,
        teardown=teardown_resources,
    )
    if pytestconfig.option.post_upgrade:
        yield secret
        secret.clean_up()
    else:
        with secret:
            yield secret


@pytest.fixture(scope="class")
def unprivileged_llama_stack_distribution_secret(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[Secret, Any, Any]:
    secret = Secret(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        name="llamastack-distribution-secret",
        type="Opaque",
        string_data=LLAMA_STACK_DISTRIBUTION_SECRET_DATA,
        ensure_exists=pytestconfig.option.post_upgrade,
        teardown=teardown_resources,
    )
    if pytestconfig.option.post_upgrade:
        yield secret
        secret.clean_up()
    else:
        with secret:
            yield secret


@pytest.fixture(scope="class")
def unprivileged_llama_stack_distribution(
    pytestconfig: pytest.Config,
    distribution_name: str,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    enabled_llama_stack_operator: DataScienceCluster,
    request: FixtureRequest,
    llama_stack_server_config: dict[str, Any],
    ci_s3_bucket_name: str,
    ci_s3_bucket_endpoint: str,
    ci_s3_bucket_region: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    teardown_resources: bool,
    unprivileged_llama_stack_distribution_secret: Secret,
    unprivileged_postgres_deployment: Deployment,
    unprivileged_postgres_service: Service,
) -> Generator[LlamaStackDistribution]:
    if pytestconfig.option.post_upgrade:
        lls_dist = LlamaStackDistribution(
            client=unprivileged_client,
            name=distribution_name,
            namespace=unprivileged_model_namespace.name,
            ensure_exists=True,
        )
        lls_dist.wait_for_status(status=LlamaStackDistribution.Status.READY, timeout=600)
        yield lls_dist
        lls_dist.clean_up()
        return

    with create_llama_stack_distribution(
        client=unprivileged_client,
        name=distribution_name,
        namespace=unprivileged_model_namespace.name,
        replicas=1,
        server=llama_stack_server_config,
        teardown=teardown_resources,
    ) as lls_dist:
        lls_dist.wait_for_status(status=LlamaStackDistribution.Status.READY, timeout=600)
        yield lls_dist


@pytest.fixture(scope="class")
def llama_stack_distribution(
    pytestconfig: pytest.Config,
    distribution_name: str,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    enabled_llama_stack_operator: DataScienceCluster,
    request: FixtureRequest,
    llama_stack_server_config: dict[str, Any],
    ci_s3_bucket_name: str,
    ci_s3_bucket_endpoint: str,
    ci_s3_bucket_region: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    teardown_resources: bool,
    llama_stack_distribution_secret: Secret,
    postgres_deployment: Deployment,
    postgres_service: Service,
) -> Generator[LlamaStackDistribution]:
    if pytestconfig.option.post_upgrade:
        lls_dist = LlamaStackDistribution(
            client=admin_client,
            name=distribution_name,
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        lls_dist.wait_for_status(status=LlamaStackDistribution.Status.READY, timeout=600)
        yield lls_dist
        lls_dist.clean_up()
        return

    with create_llama_stack_distribution(
        client=admin_client,
        name=distribution_name,
        namespace=model_namespace.name,
        replicas=1,
        server=llama_stack_server_config,
        teardown=teardown_resources,
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
    deployment.timeout_seconds = 240
    deployment.wait(timeout=240)
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
    pytestconfig: pytest.Config,
    client: DynamicClient,
    namespace: Namespace,
    deployment: Deployment,
    teardown_resources: bool,
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
    if pytestconfig.option.pre_upgrade or pytestconfig.option.post_upgrade:
        # Keep the upgrade route name short to avoid OpenShift-generated host labels
        # exceeding the DNS label limit (63 chars).
        route_name = "lls-upg-route"
        upgrade_route_patch = {
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
    else:
        route_name = generate_random_name(prefix="llama-stack", length=12)

    if pytestconfig.option.post_upgrade:
        route = Route(
            client=client,
            namespace=namespace.name,
            name=route_name,
            ensure_exists=True,
        )
        ResourceEditor(
            patches={
                route: upgrade_route_patch,
            }
        ).update()
        route.wait(timeout=60)
        yield route
        if teardown_resources:
            route.clean_up()
        return

    with Route(
        client=client,
        namespace=namespace.name,
        name=route_name,
        service=f"{deployment.name}-service",
        wait_for_resource=True,
        teardown=teardown_resources,
    ) as route:
        if pytestconfig.option.pre_upgrade:
            ResourceEditor(
                patches={
                    route: upgrade_route_patch,
                }
            ).update()
        else:
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


@pytest.fixture(scope="class")
def unprivileged_llama_stack_test_route(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    unprivileged_llama_stack_distribution_deployment: Deployment,
    teardown_resources: bool,
) -> Generator[Route, Any, Any]:
    yield from _create_llama_stack_test_route(
        pytestconfig=pytestconfig,
        client=unprivileged_client,
        namespace=unprivileged_model_namespace,
        deployment=unprivileged_llama_stack_distribution_deployment,
        teardown_resources=teardown_resources,
    )


@pytest.fixture(scope="class")
def llama_stack_test_route(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    llama_stack_distribution_deployment: Deployment,
    teardown_resources: bool,
) -> Generator[Route, Any, Any]:
    yield from _create_llama_stack_test_route(
        pytestconfig=pytestconfig,
        client=admin_client,
        namespace=model_namespace,
        deployment=llama_stack_distribution_deployment,
        teardown_resources=teardown_resources,
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
        existing_file_ids = {f.id for f in client.files.list().data}

        yield client

        _cleanup_files(client=client, existing_file_ids=existing_file_ids)
    finally:
        http_client.close()


def _cleanup_files(client: LlamaStackClient, existing_file_ids: set[str]) -> None:
    """Delete files created during test execution via the LlamaStack files API.

    Only deletes files whose IDs were not present before the test ran,
    avoiding interference with other test sessions.

    Args:
        client: The LlamaStackClient used during the test
        existing_file_ids: File IDs that existed before the test started
    """
    try:
        for file in client.files.list().data:
            if file.id not in existing_file_ids:
                try:
                    client.files.delete(file_id=file.id)
                    LOGGER.debug(f"Deleted file: {file.id}")
                except APIError as e:
                    LOGGER.warning(f"Failed to delete file {file.id}: {e}")
    except APIError as e:
        LOGGER.warning(f"Failed to clean up files: {e}")


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
    pytestconfig: pytest.Config,
    teardown_resources: bool,
) -> Generator[VectorStore]:
    """
    Creates a vector store for testing and automatically cleans it up.

    You can have example documents ingested into the store automatically by passing a
    non-empty ``doc_sources`` list in the indirect parametrization dict (URLs, files, or
    directories under the repo root). Omit ``doc_sources`` when the test only needs an
    empty store.

    Options when parametrizing with ``indirect=True``:

    * ``vector_io_provider`` (optional): backend id for the store; defaults to ``"milvus"``.
    * ``doc_sources`` (optional): non-empty list of document sources to upload after creation.
      Omitted, empty, or absent means no uploads. Each entry may be:

      * A remote URL (``http://`` or ``https://``)
      * A repo-relative or absolute file path
      * A directory path (all files in the directory are uploaded)

    Example:

        @pytest.mark.parametrize(
            "vector_store",
            [
                pytest.param(
                    {
                        "vector_io_provider": "milvus",
                        "doc_sources": [
                            "https://www.ibm.com/downloads/documents/us-en/1550f7eea8c0ded6",
                            "tests/llama_stack/dataset/corpus/finance",
                            "tests/llama_stack/dataset/corpus/finance/ibm-4q25-earnings-press-release-unencrypted.pdf",
                        ],
                    },
                    id="doc_sources:url+folder+file",
                ),
            ],
            indirect=True,
        )

    Post-upgrade runs reuse the existing store; uploads run only in the create path when
    ``doc_sources`` is non-empty (documents from the pre-upgrade run are reused otherwise).

    Args:
        unprivileged_llama_stack_client: The configured LlamaStackClient
        llama_stack_models: Model information including embedding model details
        request: Pytest fixture request carrying optional param dict
        pytestconfig: Pytest config (post-upgrade reuses store, no create/upload path)
        teardown_resources: Whether to delete the store after the class

    Yields:
        Vector store object that can be used in tests
    """

    params_raw = getattr(request, "param", None)
    params: dict[str, Any] = dict(params_raw) if isinstance(params_raw, dict) else {"vector_io_provider": "milvus"}
    vector_io_provider = str(params.get("vector_io_provider") or "milvus")
    doc_sources = params.get("doc_sources")

    if pytestconfig.option.post_upgrade:
        stores = unprivileged_llama_stack_client.vector_stores.list().data
        vector_store = next(
            (vs for vs in stores if getattr(vs, "name", "") == "test_vector_store"),
            None,
        )
        if not vector_store:
            raise ValueError("Expected vector store 'test_vector_store' to exist in post-upgrade run")
        LOGGER.info(f"Reusing existing vector_store in post-upgrade run (id={vector_store.id})")
    else:
        vector_store = unprivileged_llama_stack_client.vector_stores.create(
            name="test_vector_store",
            extra_body={
                "embedding_model": llama_stack_models.embedding_model.id,
                "embedding_dimension": llama_stack_models.embedding_dimension,
                "provider_id": vector_io_provider,
            },
        )
        LOGGER.info(f"vector_store successfully created (provider_id={vector_io_provider}, id={vector_store.id})")

        if doc_sources:
            if not isinstance(doc_sources, list):
                raise TypeError(f"doc_sources must be a list[str], got {type(doc_sources).__name__}")
            LOGGER.info(
                "Uploading doc_sources to vector_store (provider_id=%s, id=%s): %s",
                vector_io_provider,
                vector_store.id,
                doc_sources,
            )
            for source in doc_sources:
                if source.startswith(("http://", "https://")):
                    vector_store_create_file_from_url(
                        url=source,
                        llama_stack_client=unprivileged_llama_stack_client,
                        vector_store=vector_store,
                    )
                else:
                    source_path = Path(source)  # noqa: FCN001
                    if not source_path.is_absolute():
                        source_path = Path(request.config.rootdir) / source_path  # noqa: FCN001

                    if source_path.is_dir():
                        files = sorted(source_path.iterdir())
                        if not files:
                            raise FileNotFoundError(f"No files found in directory: {source_path}")
                        for file_path in files:
                            if file_path.is_file():
                                vector_store_create_file_from_path(
                                    file_path=file_path,
                                    llama_stack_client=unprivileged_llama_stack_client,
                                    vector_store=vector_store,
                                )
                    elif source_path.is_file():
                        vector_store_create_file_from_path(
                            file_path=source_path,
                            llama_stack_client=unprivileged_llama_stack_client,
                            vector_store=vector_store,
                        )
                    else:
                        raise FileNotFoundError(f"Document source not found: {source_path}")

    yield vector_store

    if teardown_resources:
        try:
            unprivileged_llama_stack_client.vector_stores.delete(vector_store_id=vector_store.id)
            LOGGER.info(f"Deleted vector store {vector_store.id}")
        except Exception as e:  # noqa: BLE001
            LOGGER.warning(f"Failed to delete vector store {vector_store.id}: {e}")


@pytest.fixture(scope="class")
def unprivileged_postgres_service(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    unprivileged_postgres_deployment: Deployment,
    teardown_resources: bool,
) -> Generator[Service, Any, Any]:
    """Create a service for the unprivileged postgres deployment."""
    service = Service(
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
        ensure_exists=pytestconfig.option.post_upgrade,
        teardown=teardown_resources,
    )
    if pytestconfig.option.post_upgrade:
        yield service
        service.clean_up()
    else:
        with service:
            yield service


@pytest.fixture(scope="class")
def unprivileged_postgres_deployment(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[Deployment, Any, Any]:
    """Deploy a Postgres instance for vector I/O provider testing with unprivileged client."""
    deployment = Deployment(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        name="vector-io-postgres-deployment",
        min_ready_seconds=5,
        replicas=1,
        selector={"matchLabels": {"app": "postgres"}},
        strategy={"type": "Recreate"},
        template=get_postgres_deployment_template(),
        teardown=teardown_resources,
        ensure_exists=pytestconfig.option.post_upgrade,
    )
    if pytestconfig.option.post_upgrade:
        deployment.wait_for_replicas(deployed=True, timeout=240)
        yield deployment
        deployment.clean_up()
    else:
        with deployment:
            deployment.wait_for_replicas(deployed=True, timeout=240)
            yield deployment


@pytest.fixture(scope="class")
def postgres_service(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    postgres_deployment: Deployment,
    teardown_resources: bool,
) -> Generator[Service, Any, Any]:
    """Create a service for the postgres deployment."""
    service = Service(
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
        ensure_exists=pytestconfig.option.post_upgrade,
        teardown=teardown_resources,
    )
    if pytestconfig.option.post_upgrade:
        yield service
        service.clean_up()
    else:
        with service:
            yield service


@pytest.fixture(scope="class")
def postgres_deployment(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[Deployment, Any, Any]:
    """Deploy a Postgres instance for vector I/O provider testing."""
    deployment = Deployment(
        client=admin_client,
        namespace=model_namespace.name,
        name="vector-io-postgres-deployment",
        min_ready_seconds=5,
        replicas=1,
        selector={"matchLabels": {"app": "postgres"}},
        strategy={"type": "Recreate"},
        template=get_postgres_deployment_template(),
        teardown=teardown_resources,
        ensure_exists=pytestconfig.option.post_upgrade,
    )
    if pytestconfig.option.post_upgrade:
        deployment.wait_for_replicas(deployed=True, timeout=240)
        yield deployment
        deployment.clean_up()
    else:
        with deployment:
            deployment.wait_for_replicas(deployed=True, timeout=240)
            yield deployment


def get_postgres_deployment_template() -> dict[str, Any]:
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
