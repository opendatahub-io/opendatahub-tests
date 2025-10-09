from typing import Generator

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.gateway import Gateway
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount

from utilities.constants import Timeout
from utilities.infra import s3_endpoint_secret
from utilities.llmd_utils import create_llmd_gateway, create_llmisvc
from utilities.llmd_constants import (
    DEFAULT_GATEWAY_NAMESPACE,
    VLLM_STORAGE_OCI,
    VLLM_STORAGE_HF,
    VLLM_CPU_IMAGE,
    DEFAULT_S3_STORAGE_PATH,
    DEFAULT_GPU_LIMIT,
    DEFAULT_GPU_REQUEST,
    DEFAULT_GPU_CPU_LIMIT,
    DEFAULT_GPU_MEMORY_LIMIT,
    DEFAULT_GPU_CPU_REQUEST,
    DEFAULT_GPU_MEMORY_REQUEST,
    DEFAULT_LLMD_REPLICAS,
)


@pytest.fixture(scope="session")
def gateway_namespace(admin_client: DynamicClient) -> str:
    return DEFAULT_GATEWAY_NAMESPACE


@pytest.fixture(scope="class")
def llmd_s3_secret(
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
) -> Generator[Secret, None, None]:
    with s3_endpoint_secret(
        client=admin_client,
        name="llmd-s3-secret",
        namespace=unprivileged_model_namespace.name,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_region=models_s3_bucket_region,
        aws_s3_bucket=models_s3_bucket_name,
        aws_s3_endpoint=models_s3_bucket_endpoint,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def llmd_s3_service_account(
    admin_client: DynamicClient, llmd_s3_secret: Secret
) -> Generator[ServiceAccount, None, None]:
    with ServiceAccount(
        client=admin_client,
        namespace=llmd_s3_secret.namespace,
        name="llmd-s3-service-account",
        secrets=[{"name": llmd_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="session")
def shared_llmd_gateway(
    admin_client: DynamicClient,
    gateway_namespace: str,
) -> Generator[Gateway, None, None]:
    """
    Session-scoped gateway shared across all LLMD tests.
    Created once at the beginning of the LLMD test session and reused.
    Uses the correct gateway class for the cluster.
    """
    # Use the correct gateway class for this cluster
    gateway_class_name = "data-science-gateway-class"

    with create_llmd_gateway(
        client=admin_client,
        namespace=gateway_namespace,
        gateway_class_name=gateway_class_name,
        wait_for_condition=True,
        timeout=Timeout.TIMEOUT_5MIN,
        teardown=True,  # Clean up at end of session
    ) as gateway:
        yield gateway


@pytest.fixture(scope="class")
def llmd_gateway(shared_llmd_gateway: Gateway) -> Gateway:
    """
    Use the shared session-scoped gateway for LLMD tests.
    No need to create/destroy gateway per test class.
    """
    return shared_llmd_gateway


@pytest.fixture(scope="class")
def llmd_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[LLMInferenceService, None, None]:
    if isinstance(request.param, str):
        name_suffix = request.param
        kwargs = {}
    else:
        name_suffix = request.param.get("name_suffix", "basic")
        kwargs = {k: v for k, v in request.param.items() if k != "name_suffix"}

    service_name = kwargs.get("name", f"llm-{name_suffix}")

    if "llmd_gateway" in request.fixturenames:
        request.getfixturevalue(argname="llmd_gateway")
    container_resources = kwargs.get(
        "container_resources",
        {
            "limits": {"cpu": "2", "memory": "16Gi"},
            "requests": {"cpu": "500m", "memory": "12Gi"},
        },
    )

    # Allow users to specify custom image, fallback to CPU default if needed
    create_kwargs = {
        "client": admin_client,
        "name": service_name,
        "namespace": unprivileged_model_namespace.name,
        "storage_uri": kwargs.get("storage_uri", VLLM_STORAGE_OCI),
        "container_image": kwargs.get("container_image", VLLM_CPU_IMAGE),
        "container_resources": container_resources,
        "wait": True,
        "timeout": Timeout.TIMEOUT_15MIN,
        **{k: v for k, v in kwargs.items() if k != "name"},
    }

    with create_llmisvc(**create_kwargs) as llm_service:
        yield llm_service


@pytest.fixture(scope="class")
def llmd_inference_service_s3(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    llmd_s3_secret: Secret,
    llmd_s3_service_account: ServiceAccount,
) -> Generator[LLMInferenceService, None, None]:
    if isinstance(request.param, str):
        name_suffix = request.param
        kwargs = {"storage_path": DEFAULT_S3_STORAGE_PATH}
    else:
        name_suffix = request.param.get("name_suffix", "s3")
        kwargs = {k: v for k, v in request.param.items() if k != "name_suffix"}

    service_name = kwargs.get("name", f"llm-{name_suffix}")

    if "storage_key" not in kwargs:
        kwargs["storage_key"] = llmd_s3_secret.name

    if "storage_path" not in kwargs:
        kwargs["storage_path"] = DEFAULT_S3_STORAGE_PATH

    container_resources = kwargs.get(
        "container_resources",
        {
            "limits": {"cpu": "1", "memory": "10Gi"},
            "requests": {"cpu": "100m", "memory": "8Gi"},
        },
    )

    # Allow users to specify custom image, fallback to CPU default if needed
    create_kwargs = {
        "client": admin_client,
        "name": service_name,
        "namespace": unprivileged_model_namespace.name,
        "storage_key": kwargs.get("storage_key"),
        "storage_path": kwargs.get("storage_path"),
        "container_image": kwargs.get("container_image", VLLM_CPU_IMAGE),
        "container_resources": container_resources,
        "service_account": llmd_s3_service_account.name,
        "wait": True,
        "timeout": Timeout.TIMEOUT_15MIN,
        **{
            k: v
            for k, v in kwargs.items()
            if k not in ["name", "storage_key", "storage_path", "container_image", "container_resources"]
        },
    }

    with create_llmisvc(**create_kwargs) as llm_service:
        yield llm_service


@pytest.fixture(scope="class")
def llmd_inference_service_gpu(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[LLMInferenceService, None, None]:
    if isinstance(request.param, str):
        name_suffix = request.param
        kwargs = {}
    else:
        name_suffix = request.param.get("name_suffix", "gpu-hf")
        kwargs = {k: v for k, v in request.param.items() if k != "name_suffix"}

    service_name = kwargs.get("name", f"llm-{name_suffix}")

    if "llmd_gateway" in request.fixturenames:
        request.getfixturevalue(argname="llmd_gateway")

    # GPU resource configuration (reduce for prefill/decode with smaller model)
    if kwargs.get("enable_prefill_decode", False):
        # Reduced resources for prefill/decode with TinyLlama
        container_resources = kwargs.get(
            "container_resources",
            {
                "limits": {"cpu": "2", "memory": "16Gi", "nvidia.com/gpu": "1"},
                "requests": {"cpu": "1", "memory": "8Gi", "nvidia.com/gpu": "1"},
            },
        )
    else:
        # Standard GPU resource configuration (matches temp YAML files)
        container_resources = kwargs.get(
            "container_resources",
            {
                "limits": {
                    "cpu": DEFAULT_GPU_CPU_LIMIT,
                    "memory": DEFAULT_GPU_MEMORY_LIMIT,
                    "nvidia.com/gpu": DEFAULT_GPU_LIMIT,
                },
                "requests": {
                    "cpu": DEFAULT_GPU_CPU_REQUEST,
                    "memory": DEFAULT_GPU_MEMORY_REQUEST,
                    "nvidia.com/gpu": DEFAULT_GPU_REQUEST,
                },
            },
        )

    # Configure liveness probe (from temp YAML files)
    liveness_probe = {
        "httpGet": {"path": "/health", "port": 8000, "scheme": "HTTPS"},
        "initialDelaySeconds": 120,
        "periodSeconds": 30,
        "timeoutSeconds": 30,
        "failureThreshold": 5,
    }

    # Support variable replicas based on configuration type
    replicas = kwargs.get("replicas", DEFAULT_LLMD_REPLICAS)
    if kwargs.get("enable_prefill_decode", False):
        replicas = kwargs.get("replicas", 3)  # Default to 3 for prefill/decode as per temp YAML

    # Build minimal prefill configuration - let config reference handle details
    prefill_config = None
    if kwargs.get("enable_prefill_decode", False):
        prefill_config = {
            "replicas": kwargs.get("prefill_replicas", 1),  # Use parameter or default to 1 for cluster constraints
        }

    # Base configuration following temp YAML structure and supported parameters
    create_kwargs = {
        "client": admin_client,
        "name": service_name,
        "namespace": unprivileged_model_namespace.name,
        "storage_uri": kwargs.get("storage_uri", VLLM_STORAGE_HF),
        "model_name": kwargs.get("model_name", "Qwen/Qwen2.5-7B-Instruct"),
        "replicas": replicas,
        "container_resources": container_resources,
        "liveness_probe": liveness_probe,
        "prefill_config": prefill_config,
        "disable_scheduler": kwargs.get("disable_scheduler", False),
        "enable_prefill_decode": kwargs.get("enable_prefill_decode", False),
        "wait": True,
        "timeout": Timeout.TIMEOUT_15MIN,
    }

    # Only set container_image if explicitly provided by user (follows temp YAML pattern)
    if "container_image" in kwargs:
        create_kwargs["container_image"] = kwargs["container_image"]

    with create_llmisvc(**create_kwargs) as llm_service:
        yield llm_service
