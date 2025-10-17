from contextlib import ExitStack
from typing import Generator

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount

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
def deepseek_r1_inference_service(
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[LLMInferenceService, None, None]:
    """Fixture for DeepSeek R1 0528 model with multi-node configuration."""
    service_name = "deepseek-r1-0528"

    # Define common environment variables for both template and worker
    common_env = [
        {"name": "VLLM_LOGGING_LEVEL", "value": "INFO"},
        {"name": "KSERVE_INFER_ROCE", "value": "true"},
        {"name": "CUDA_DEVICE_ORDER", "value": "PCI_BUS_ID"},
        # Memory optimizations
        {"name": "VLLM_ADDITIONAL_ARGS", "value": "--gpu-memory-utilization 0.95 --max-model-len 8192 --enforce-eager"},
        {"name": "VLLM_ALL2ALL_BACKEND", "value": "deepep_high_throughput"},
        {"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "expandable_segments:True"},
        # Essential NCCL configuration
        {"name": "NCCL_IB_GID_INDEX", "value": "3"},
        {"name": "NCCL_DEBUG", "value": "WARN"},
        {"name": "NCCL_SOCKET_IFNAME", "value": "net1"},
        {"name": "NCCL_IB_TIMEOUT", "value": "100"},
        # NVSHMEM configuration - optimized for stability
        {"name": "NVSHMEM_REMOTE_TRANSPORT", "value": "ibgda"},
        {"name": "NVSHMEM_BOOTSTRAP_TWO_STAGE", "value": "1"},
        {"name": "NVSHMEM_BOOTSTRAP_TIMEOUT", "value": "300"},
        {"name": "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME", "value": "net1"},
        {"name": "NVSHMEM_IB_GID_INDEX", "value": "3"},
        {"name": "NVSHMEM_USE_IBGDA", "value": "1"},
        {"name": "NVSHMEM_ENABLE_NIC_PE_MAPPING", "value": "1"},
        {"name": "NVSHMEM_IBGDA_SUPPORT", "value": "1"},
        {"name": "NVSHMEM_IB_ENABLE_IBGDA", "value": "1"},
        {"name": "NVSHMEM_IBGDA_NIC_HANDLER", "value": "gpu"},
        {"name": "NVSHMEM_DEBUG", "value": "WARN"},
        # UCX configuration for NVSHMEM
        {"name": "UCX_TLS", "value": "rc,sm,self,cuda_copy,cuda_ipc"},
        {"name": "UCX_IB_GID_INDEX", "value": "3"},
        {"name": "UCX_RC_MLX5_TM_ENABLE", "value": "n"},
        {"name": "UCX_UD_MLX5_RX_QUEUE_LEN", "value": "1024"},
        {"name": "NVIDIA_GDRCOPY", "value": "enabled"},
    ]

    container_resources = {
        "limits": {
            "cpu": "128",
            "ephemeral-storage": "800Gi",
            "memory": "512Gi",
            "nvidia.com/gpu": "8",
            "rdma/roce_gdr": "1",
        },
        "requests": {
            "cpu": "64",
            "ephemeral-storage": "800Gi",
            "memory": "256Gi",
            "nvidia.com/gpu": "8",
            "rdma/roce_gdr": "1",
        },
    }

    liveness_probe = {
        "httpGet": {"path": "/health", "port": 8000, "scheme": "HTTPS"},
        "initialDelaySeconds": 4800,
        "periodSeconds": 10,
        "timeoutSeconds": 10,
        "failureThreshold": 3,
    }

    parallelism_config = {
        "data": 32,
        "dataLocal": 8,
        "expert": True,
        "tensor": 1,
    }

    router_config = {
        "scheduler": {},
        "route": {},
        "gateway": {},
    }

    worker_spec = {
        "serviceAccountName": "hfsa",
        "containers": [
            {
                "name": "main",
                "env": common_env,
                "resources": container_resources,
            }
        ],
    }

    annotations = {
        "k8s.v1.cni.cncf.io/networks": "roce-p2",
    }

    with create_llmisvc(
        client=admin_client,
        name=service_name,
        namespace=unprivileged_model_namespace.name,
        storage_uri=ModelStorage.HF_DEEPSEEK_R1_0528,
        model_name="deepseek-ai/DeepSeek-R1-0528",
        replicas=1,
        parallelism=parallelism_config,
        router_config=router_config,
        container_env=common_env,
        container_resources=container_resources,
        liveness_probe=liveness_probe,
        service_account="hfsa",
        worker_config=worker_spec,
        annotations=annotations,
        wait=True,
        timeout=Timeout.TIMEOUT_30MIN,
    ) as llm_service:
        yield llm_service
