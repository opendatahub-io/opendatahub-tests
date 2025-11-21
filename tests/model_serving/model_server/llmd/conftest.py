from contextlib import ExitStack
from typing import Generator, Literal, Any

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount

from tests.model_serving.model_server.llmd.constants import (
    MULTINODE_LIVENESS_PROBE,
    MULTINODE_SCHEDULER_CONFIG_PD,
    SINGLENODE_LIVENESS_PROBE,
    SINGLENODE_SCHEDULER_CONFIG_PD,
    ROCE_ANNOTATION,
)
from tests.model_serving.model_server.llmd.utils import (
    get_kv_transfer_env,
    get_scheduler_container_args,
    get_common_multinode_env,
    build_scheduler_router_config,
    build_prefill_config,
)
from utilities.constants import Timeout, ResourceLimits
from utilities.infra import s3_endpoint_secret, create_inference_token
from utilities.logger import RedactedString
from utilities.llmd_utils import create_llmisvc
from utilities.llmd_constants import (
    ModelStorage,
    ContainerImages,
    ModelNames,
    LLMDDefaults,
)


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
        kwargs = {}
    else:
        name_suffix = request.param.get("name_suffix", "s3")
        kwargs = {k: v for k, v in request.param.items() if k != "name_suffix"}

    service_name = kwargs.get("name", f"llm-{name_suffix}")

    container_resources = kwargs.get(
        "container_resources",
        {
            "limits": {"cpu": "1", "memory": "10Gi"},
            "requests": {"cpu": "100m", "memory": "8Gi"},
        },
    )

    create_kwargs = {
        "client": admin_client,
        "name": service_name,
        "namespace": unprivileged_model_namespace.name,
        "storage_uri": kwargs.get("storage_uri", ModelStorage.TINYLLAMA_S3),
        "container_image": kwargs.get("container_image", ContainerImages.VLLM_CPU),
        "container_resources": container_resources,
        "service_account": llmd_s3_service_account.name,
        "wait": True,
        "timeout": Timeout.TIMEOUT_15MIN,
        **{
            k: v
            for k, v in kwargs.items()
            if k not in ["name", "storage_uri", "container_image", "container_resources"]
        },
    }

    with create_llmisvc(**create_kwargs) as llm_service:
        yield llm_service


@pytest.fixture(scope="class")
def llmd_inference_service_gpu(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    llmd_s3_secret: Secret,
    llmd_s3_service_account: ServiceAccount,
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

    if kwargs.get("enable_prefill_decode", False):
        container_resources = kwargs.get(
            "container_resources",
            {
                "limits": {"cpu": "4", "memory": "32Gi", "nvidia.com/gpu": "1"},
                "requests": {"cpu": "2", "memory": "16Gi", "nvidia.com/gpu": "1"},
            },
        )
    else:
        container_resources = kwargs.get(
            "container_resources",
            {
                "limits": {
                    "cpu": ResourceLimits.GPU.CPU_LIMIT,
                    "memory": ResourceLimits.GPU.MEMORY_LIMIT,
                    "nvidia.com/gpu": ResourceLimits.GPU.LIMIT,
                },
                "requests": {
                    "cpu": ResourceLimits.GPU.CPU_REQUEST,
                    "memory": ResourceLimits.GPU.MEMORY_REQUEST,
                    "nvidia.com/gpu": ResourceLimits.GPU.REQUEST,
                },
            },
        )

    liveness_probe = {
        "httpGet": {"path": "/health", "port": 8000, "scheme": "HTTPS"},
        "initialDelaySeconds": 120,
        "periodSeconds": 30,
        "timeoutSeconds": 30,
        "failureThreshold": 5,
    }

    replicas = kwargs.get("replicas", LLMDDefaults.REPLICAS)
    if kwargs.get("enable_prefill_decode", False):
        replicas = kwargs.get("replicas", 3)

    prefill_config = None
    if kwargs.get("enable_prefill_decode", False):
        prefill_config = {
            "replicas": kwargs.get("prefill_replicas", 1),
        }

    create_kwargs = {
        "client": admin_client,
        "name": service_name,
        "namespace": unprivileged_model_namespace.name,
        "storage_uri": kwargs.get("storage_uri", ModelStorage.S3_QWEN),
        "model_name": kwargs.get("model_name", ModelNames.QWEN),
        "replicas": replicas,
        "container_resources": container_resources,
        "liveness_probe": liveness_probe,
        "prefill_config": prefill_config,
        "disable_scheduler": kwargs.get("disable_scheduler", False),
        "enable_prefill_decode": kwargs.get("enable_prefill_decode", False),
        "service_account": llmd_s3_service_account.name,
        "wait": True,
        "timeout": Timeout.TIMEOUT_15MIN,
    }

    if "container_image" in kwargs:
        create_kwargs["container_image"] = kwargs["container_image"]

    with create_llmisvc(**create_kwargs) as llm_service:
        yield llm_service


@pytest.fixture(scope="class")
def llmisvc_auth_service_account(
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator:
    """Factory fixture to create service accounts for authentication testing."""
    with ExitStack() as stack:

        def _create_service_account(name: str) -> ServiceAccount:
            """Create a single service account."""
            return stack.enter_context(
                cm=ServiceAccount(
                    client=admin_client,
                    namespace=unprivileged_model_namespace.name,
                    name=name,
                )
            )

        yield _create_service_account


@pytest.fixture(scope="class")
def llmisvc_auth_view_role(
    admin_client: DynamicClient,
) -> Generator:
    """Factory fixture to create view roles for LLMInferenceServices."""
    with ExitStack() as stack:

        def _create_view_role(llm_service: LLMInferenceService) -> Role:
            """Create a single view role for a given LLMInferenceService."""
            return stack.enter_context(
                cm=Role(
                    client=admin_client,
                    name=f"{llm_service.name}-view",
                    namespace=llm_service.namespace,
                    rules=[
                        {
                            "apiGroups": [llm_service.api_group],
                            "resources": ["llminferenceservices"],
                            "verbs": ["get"],
                            "resourceNames": [llm_service.name],
                        },
                    ],
                )
            )

        yield _create_view_role


@pytest.fixture(scope="class")
def llmisvc_auth_role_binding(
    admin_client: DynamicClient,
) -> Generator:
    """Factory fixture to create role bindings."""
    with ExitStack() as stack:

        def _create_role_binding(
            service_account: ServiceAccount,
            role: Role,
        ) -> RoleBinding:
            """Create a single role binding."""
            return stack.enter_context(
                cm=RoleBinding(
                    client=admin_client,
                    namespace=service_account.namespace,
                    name=f"{service_account.name}-view",
                    role_ref_name=role.name,
                    role_ref_kind=role.kind,
                    subjects_kind="ServiceAccount",
                    subjects_name=service_account.name,
                )
            )

        yield _create_role_binding


@pytest.fixture(scope="class")
def llmisvc_auth_token() -> Generator:
    """Factory fixture to create inference tokens with all required RBAC resources."""

    def _create_token(
        service_account: ServiceAccount,
        llmisvc: LLMInferenceService,
        view_role_factory,
        role_binding_factory,
    ) -> str:
        """Create role, role binding, and return an inference token for an existing service account."""
        # Create role and role binding (these factories manage their own cleanup via ExitStack)
        role = view_role_factory(llm_service=llmisvc)
        role_binding_factory(service_account=service_account, role=role)
        return RedactedString(value=create_inference_token(model_service_account=service_account))

    yield _create_token


@pytest.fixture(scope="class")
def llmisvc_auth(
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    llmisvc_auth_service_account,
) -> Generator:
    """Factory fixture to create LLMInferenceService instances for authentication testing."""
    with ExitStack() as stack:

        def _create_llmd_auth_service(
            service_name: str,
            service_account_name: str,
            storage_uri: str = ModelStorage.TINYLLAMA_OCI,
            container_image: str = ContainerImages.VLLM_CPU,
            container_resources: dict | None = None,
        ) -> tuple[LLMInferenceService, ServiceAccount]:
            """Create a single LLMInferenceService instance with its service account."""
            if container_resources is None:
                container_resources = {
                    "limits": {"cpu": "1", "memory": "10Gi"},
                    "requests": {"cpu": "100m", "memory": "8Gi"},
                }

            # Create the service account first
            sa = llmisvc_auth_service_account(name=service_account_name)

            create_kwargs = {
                "client": admin_client,
                "name": service_name,
                "namespace": unprivileged_model_namespace.name,
                "storage_uri": storage_uri,
                "container_image": container_image,
                "container_resources": container_resources,
                "service_account": service_account_name,
                "wait": True,
                "timeout": Timeout.TIMEOUT_15MIN,
                "enable_auth": True,
            }

            llm_service = stack.enter_context(cm=create_llmisvc(**create_kwargs))
            return (llm_service, sa)

        yield _create_llmd_auth_service


@pytest.fixture(scope="class")
def _create_multinode_llmisvc(
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    *,
    service_name: str,
    storage_uri: str,
    model_name: str,
    mode: Literal["rdma", "tcp", "rdma_pd"],
    data_parallelism: int,
    gpu_count: int | str = "8",
    gpu_memory_util: str = "0.95",
    max_model_len: str = "8192",
    enable_kv_transfer: bool = False,
    prefill_config: dict[str, Any] | None = None,
    scheduler_config_text: str | None = None,
    scheduler_verbose_level: str = "",
    service_account: str = "hfsa",
    **kwargs,
) -> Generator[LLMInferenceService, None, None]:
    """Factory for creating multinode LLMISVC with common logic.

    Args:
        admin_client: Kubernetes dynamic client
        unprivileged_model_namespace: Target namespace
        service_name: Name of the LLM service
        storage_uri: Model storage URI
        model_name: Model identifier
        mode: Network configuration mode ("rdma", "tcp", or "rdma_pd")
        data_parallelism: Data parallelism count
        gpu_count: Number of GPUs per node (default: "8")
        gpu_memory_util: GPU memory utilization (default: "0.95")
        max_model_len: Maximum model length (default: "8192")
        enable_kv_transfer: Enable KV cache transfer (default: False)
        prefill_config: Optional prefill configuration dict
        scheduler_config_text: Optional scheduler YAML config
        scheduler_verbose_level: Scheduler verbosity (e.g., "-v=4")
        service_account: Service account name (default: "hfsa")
        **kwargs: Additional arguments passed to create_llmisvc

    Yields:
        LLMInferenceService instance
    """
    # Get environment variables based on mode
    common_env = get_common_multinode_env(
        mode=mode,
        gpu_memory_util=gpu_memory_util,
        max_model_len=max_model_len,
        enable_kv_transfer=enable_kv_transfer,
    )

    # Build container resources
    container_resources = {
        "limits": {
            "cpu": "128",
            "ephemeral-storage": "800Gi" if mode in ("rdma", "rdma_pd") else "100Gi",
            "memory": "512Gi",
            "nvidia.com/gpu": str(gpu_count),
        },
        "requests": {
            "cpu": "64",
            "ephemeral-storage": "800Gi" if mode in ("rdma", "rdma_pd") else "100Gi",
            "memory": "256Gi",
            "nvidia.com/gpu": str(gpu_count),
        },
    }

    # Add RDMA resources for RDMA modes
    if mode in ("rdma", "rdma_pd"):
        container_resources["limits"]["rdma/roce_gdr"] = 1
        container_resources["requests"]["rdma/roce_gdr"] = 1

    # Build parallelism config
    parallelism_config = {
        "data": data_parallelism,
        "dataLocal": 8,
        "expert": True,
        "tensor": 1,
    }

    # Build router config
    router_config: dict[str, Any] = {
        "scheduler": {},
        "route": {},
        "gateway": {},
    }

    # Add scheduler config if provided
    if scheduler_config_text:
        router_config["scheduler"] = {
            "template": {
                "containers": [
                    {
                        "name": "main",
                        "args": get_scheduler_container_args(
                            scheduler_config_text,
                            verbose_level=scheduler_verbose_level,
                        ),
                    }
                ]
            }
        }

    # Build worker spec
    worker_spec: dict[str, Any] = {
        "containers": [
            {
                "name": "main",
                "env": common_env,
                "resources": container_resources,
            }
        ],
    }

    # Add service account for RDMA modes or if explicitly provided
    if mode in ("rdma", "rdma_pd") or service_account:
        worker_spec["serviceAccountName"] = service_account

    # Determine annotations based on mode
    if mode in ("rdma", "rdma_pd"):
        annotations = ROCE_ANNOTATION
    else:  # mode == "tcp"
        annotations = {"security.opendatahub.io/enable-network-policies": "false"}

    with create_llmisvc(
        client=admin_client,
        name=service_name,
        namespace=unprivileged_model_namespace.name,
        storage_uri=storage_uri,
        model_name=model_name,
        replicas=1,
        parallelism=parallelism_config,
        router_config=router_config,
        container_env=common_env,
        container_resources=container_resources,
        liveness_probe=MULTINODE_LIVENESS_PROBE,
        worker_config=worker_spec,
        service_account=service_account if mode in ("rdma", "rdma_pd") else None,
        prefill_config=prefill_config,
        annotations=annotations,
        wait=True,
        timeout=Timeout.TIMEOUT_30MIN,
        **kwargs,
    ) as llm_service:
        yield llm_service


@pytest.fixture(scope="class")
def llmisvc_multinode_dp_ep(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[LLMInferenceService, None, None]:
    """Fixture for DeepSeek R1 with multi-node DP+EP configuration (RDMA-enabled).

    Matches: multi-node-dp-ep.yaml
    - Data parallelism: 32
    - RDMA/RoCE networking
    - IBGDa transport
    """
    params = getattr(request, "param", {})
    if not isinstance(params, dict):
        params = {}

    yield from _create_multinode_llmisvc(
        admin_client=admin_client,
        unprivileged_model_namespace=unprivileged_model_namespace,
        service_name=params.get("service_name", "deepseek-r1-0528"),
        storage_uri=params.get("storage_uri", "hf://deepseek-ai/DeepSeek-R1-0528"),
        model_name=params.get("model_name", "deepseek-ai/DeepSeek-R1-0528"),
        mode="rdma",
        data_parallelism=32,
        gpu_count="8",
    )


@pytest.fixture(scope="class")
def llmisvc_multinode_dp_ep_tcp(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[LLMInferenceService, None, None]:
    """Fixture for DeepSeek Coder V2 with multi-node DP+EP configuration (TCP-only).

    Matches: multi-node-dp-ep-tcp.yaml
    - Data parallelism: 16
    - TCP networking (GCP gVNIC)
    - No RDMA - uses 6 GPUs instead of 8 for cost optimization
    """
    params = getattr(request, "param", {})
    if not isinstance(params, dict):
        params = {}

    yield from _create_multinode_llmisvc(
        admin_client=admin_client,
        unprivileged_model_namespace=unprivileged_model_namespace,
        service_name=params.get("service_name", "deepseek-coder-v2"),
        storage_uri=params.get("storage_uri", "hf://deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"),
        model_name=params.get("model_name", "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"),
        mode="tcp",
        data_parallelism=16,
        gpu_count="6",  # TCP uses 6 GPUs for cost optimization on GCP
    )


@pytest.fixture(scope="class")
def llmisvc_multinode_dp_ep_prefill_decode(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[LLMInferenceService, None, None]:
    """Fixture for DeepSeek R1 with multi-node DP+EP and prefill-decode separation.

    Matches: multi-node-dp-ep-prefill-decode.yaml
    - Data parallelism: 16 (both decode and prefill)
    - RDMA/RoCE networking
    - KV cache transfer via NixlConnector
    - Advanced scheduler with PD separation
    - Decode uses 0.99 GPU memory, prefill uses 0.97
    """
    params = getattr(request, "param", {})
    if not isinstance(params, dict):
        params = {}

    # Build prefill environment with different GPU memory utilization (0.97 vs 0.99 for decode)
    prefill_env = get_common_multinode_env(
        mode="rdma_pd",
        gpu_memory_util="0.97",  # Prefill uses slightly less GPU memory
        max_model_len="4096",
        enable_kv_transfer=True,
    )

    # Build container resources (shared by both decode and prefill)
    container_resources = {
        "limits": {
            "cpu": "128",
            "ephemeral-storage": "800Gi",
            "memory": "512Gi",
            "nvidia.com/gpu": "8",
            "rdma/roce_gdr": 1,
        },
        "requests": {
            "cpu": "64",
            "ephemeral-storage": "800Gi",
            "memory": "256Gi",
            "nvidia.com/gpu": "8",
            "rdma/roce_gdr": 1,
        },
    }

    # Build parallelism config (shared by both decode and prefill)
    parallelism_config = {
        "data": 16,
        "dataLocal": 8,
        "expert": True,
        "tensor": 1,
    }

    # Build prefill configuration using helper
    prefill_config = build_prefill_config(
        replicas=1,
        env=prefill_env,
        resources=container_resources,
        liveness_probe=MULTINODE_LIVENESS_PROBE,
        parallelism=parallelism_config,
        service_account="hfsa",
    )

    yield from _create_multinode_llmisvc(
        admin_client=admin_client,
        unprivileged_model_namespace=unprivileged_model_namespace,
        service_name=params.get("service_name", "deepseek-r1-0528-pd"),
        storage_uri=params.get("storage_uri", "hf://deepseek-ai/DeepSeek-R1-0528"),
        model_name=params.get("model_name", "deepseek-ai/DeepSeek-R1-0528"),
        mode="rdma_pd",
        data_parallelism=16,
        gpu_count="8",
        gpu_memory_util="0.99",  # Decode uses higher GPU memory utilization
        max_model_len="4096",
        enable_kv_transfer=True,
        prefill_config=prefill_config,
        scheduler_config_text=MULTINODE_SCHEDULER_CONFIG_PD,
        scheduler_verbose_level="-v=4",
    )


@pytest.fixture(scope="class")
def llmisvc_singlenode_prefill_decode(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[LLMInferenceService, None, None]:
    """Fixture for single-node GPU LLMInferenceService with prefill-decode separation."""
    params = getattr(request, "param", {})
    if not isinstance(params, dict):
        params = {}

    service_name = params.get("service_name", "qwen2-7b-instruct-pd")
    storage_uri = params.get("storage_uri", "hf://Qwen/Qwen2.5-7B-Instruct")
    model_name = params.get("model_name", "Qwen/Qwen2.5-7B-Instruct")
    decode_replicas = params.get("decode_replicas", 1)
    prefill_replicas = params.get("prefill_replicas", 2)

    # Environment variables for KV cache transfer via RDMA
    common_env = get_kv_transfer_env()

    container_resources = {
        "limits": {
            "cpu": "4",
            "memory": "32Gi",
            "nvidia.com/gpu": "1",
            "rdma/roce_gdr": "1",
        },
        "requests": {
            "cpu": "2",
            "memory": "16Gi",
            "nvidia.com/gpu": "1",
            "rdma/roce_gdr": "1",
        },
    }

    # Build router config with scheduler using helper
    router_config = build_scheduler_router_config(scheduler_config_text=SINGLENODE_SCHEDULER_CONFIG_PD)

    # Build prefill configuration using helper
    prefill_config = build_prefill_config(
        replicas=prefill_replicas,
        env=common_env,
        resources=container_resources,
        liveness_probe=SINGLENODE_LIVENESS_PROBE,
    )

    with create_llmisvc(
        client=admin_client,
        name=service_name,
        namespace=unprivileged_model_namespace.name,
        storage_uri=storage_uri,
        model_name=model_name,
        replicas=decode_replicas,
        router_config=router_config,
        container_env=common_env,
        container_resources=container_resources,
        liveness_probe=SINGLENODE_LIVENESS_PROBE,
        prefill_config=prefill_config,
        annotations=ROCE_ANNOTATION,
        wait=True,
        timeout=Timeout.TIMEOUT_30MIN,
    ) as llm_service:
        yield llm_service
