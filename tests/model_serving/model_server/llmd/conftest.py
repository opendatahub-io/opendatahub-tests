from collections.abc import Generator
from contextlib import ExitStack

import pytest
import yaml
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.gateway import Gateway
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount

from tests.model_serving.model_server.llmd.constants import (
    LLMD_LIVENESS_PROBE,
    PREFIX_CACHE_BLOCK_SIZE,
    PREFIX_CACHE_HASH_ALGO,
    PREFIX_CACHE_HASH_SEED,
    ROUTER_SCHEDULER_CONFIG_ESTIMATED_PREFIX_CACHE,
    ROUTER_SCHEDULER_CONFIG_PRECISE_PREFIX_CACHE,
)
from utilities.constants import ResourceLimits, Timeout
from utilities.infra import create_inference_token, s3_endpoint_secret
from utilities.llmd_constants import (
    ContainerImages,
    LLMDDefaults,
    LLMDGateway,
    ModelNames,
    ModelStorage,
)
from utilities.llmd_utils import create_llmd_gateway, create_llmisvc
from utilities.logger import RedactedString


# *********************************
# **         S3 fixtures         **
# *********************************
@pytest.fixture(scope="class")
def llmd_s3_secret(
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
) -> Generator[Secret]:
    """Create a Kubernetes secret with S3 credentials for LLMD model storage."""
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
def llmd_s3_service_account(admin_client: DynamicClient, llmd_s3_secret: Secret) -> Generator[ServiceAccount]:
    """Create a service account linked to the S3 secret for LLMD pods."""
    with ServiceAccount(
        client=admin_client,
        namespace=llmd_s3_secret.namespace,
        name="llmd-s3-service-account",
        secrets=[{"name": llmd_s3_secret.name}],
    ) as sa:
        yield sa


# *********************************
# **      Gateway fixtures       **
# *********************************
@pytest.fixture(scope="session")
def gateway_namespace() -> str:
    """Return the namespace for LLMD gateway."""
    return LLMDGateway.DEFAULT_NAMESPACE


@pytest.fixture(scope="session")
def shared_llmd_gateway(
    admin_client: DynamicClient,
    gateway_namespace: str,
) -> Generator[Gateway]:
    """Shared LLMD gateway for all tests."""
    with create_llmd_gateway(
        client=admin_client,
        namespace=gateway_namespace,
        gateway_class_name=LLMDGateway.DEFAULT_CLASS,
        wait_for_condition=True,
        timeout=Timeout.TIMEOUT_5MIN,
        teardown=True,
    ) as gateway:
        yield gateway


@pytest.fixture(scope="class")
def llmd_gateway(shared_llmd_gateway: Gateway) -> Gateway:
    """Class-scoped LLMD gateway fixture."""
    return shared_llmd_gateway


# *********************************
# **        Auth fixtures        **
# *********************************
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
def authenticated_llmisvc_token(
    request: FixtureRequest,
    llmisvc_auth_token,
    llmisvc_auth_view_role,
    llmisvc_auth_role_binding,
) -> str:
    """Create an authentication token for accessing a protected LLMInferenceService."""
    service_account_fixture_name = request.param["service_account_fixture"]
    llmisvc_fixture_name = request.param["llmisvc_fixture"]

    # Get fixtures dynamically
    service_account = request.getfixturevalue(argname=service_account_fixture_name)
    llmisvc = request.getfixturevalue(argname=llmisvc_fixture_name)

    # Create and return token
    return llmisvc_auth_token(
        service_account=service_account,
        llmisvc=llmisvc,
        view_role_factory=llmisvc_auth_view_role,
        role_binding_factory=llmisvc_auth_role_binding,
    )


# ************************************
# ** LLM Inference Service fixtures **
# ************************************
@pytest.fixture(scope="class")
def llmd_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[LLMInferenceService]:
    """Basic LLMInferenceService fixture for OCI storage with CPU runtime.

    This is the most commonly used fixture for basic LLMD tests. It uses
    OCI container registry for model storage and defaults to CPU resources.
    """
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

    create_kwargs = {
        "client": admin_client,
        "name": service_name,
        "namespace": unprivileged_model_namespace.name,
        "storage_uri": kwargs.get("storage_uri", ModelStorage.TINYLLAMA_OCI),
        "container_image": kwargs.get("container_image", ContainerImages.VLLM_CPU),
        "container_resources": container_resources,
        "liveness_probe": LLMD_LIVENESS_PROBE,
        "wait": True,
        "timeout": Timeout.TIMEOUT_15MIN,
        **{k: v for k, v in kwargs.items() if k != "name"},
    }

    with create_llmisvc(**create_kwargs) as llm_service:
        yield llm_service


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
                "liveness_probe": LLMD_LIVENESS_PROBE,
                "service_account": service_account_name,
                "wait": True,
                "timeout": Timeout.TIMEOUT_15MIN,
                "enable_auth": True,
            }

            llm_service = stack.enter_context(cm=create_llmisvc(**create_kwargs))
            return (llm_service, sa)

        yield _create_llmd_auth_service


@pytest.fixture(scope="class")
def llmd_inference_service_s3(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    llmd_s3_service_account: ServiceAccount,
) -> Generator[LLMInferenceService]:
    """Create an LLMInferenceService that loads models from S3 storage."""
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
        "liveness_probe": LLMD_LIVENESS_PROBE,
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
    llmd_s3_service_account: ServiceAccount,
) -> Generator[LLMInferenceService]:
    """Create an LLMInferenceService with GPU resources for accelerated inference."""
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
        "liveness_probe": LLMD_LIVENESS_PROBE,
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
def singlenode_estimated_prefix_cache(
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    llmd_s3_secret: Secret,
    llmd_s3_service_account: ServiceAccount,
    llmd_gateway: Gateway,
) -> Generator[LLMInferenceService]:
    """LLMInferenceService fixture for single-node estimated prefix cache test."""

    llmisvc_name = "singlenode-estimated-prefix-cache"

    with create_llmisvc(
        client=admin_client,
        name=llmisvc_name,
        namespace=unprivileged_model_namespace.name,
        storage_uri=ModelStorage.TINYLLAMA_S3,
        model_name=ModelNames.TINYLLAMA,
        replicas=2,
        annotations={
            "prometheus.io/port": "8000",
            "prometheus.io/path": "/metrics",
        },
        container_resources={
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
        container_env=[
            {"name": "MODEL_NAME", "value": ModelNames.TINYLLAMA},
            {"name": "VLLM_LOGGING_LEVEL", "value": "DEBUG"},
            {"name": "PYTHONHASHSEED", "value": str(PREFIX_CACHE_HASH_SEED)},
            {
                "name": "VLLM_ADDITIONAL_ARGS",
                "value": (
                    f"--prefix-caching-hash-algo {PREFIX_CACHE_HASH_ALGO} --block-size {PREFIX_CACHE_BLOCK_SIZE}"
                ),
            },
        ],
        liveness_probe=LLMD_LIVENESS_PROBE,
        service_account=llmd_s3_service_account.name,
        enable_auth=True,
        router_config={
            "scheduler": {
                "template": {
                    "containers": [
                        {
                            "name": "main",
                            "args": [
                                "--v=4",
                                "--pool-name",
                                "{{ ChildName .ObjectMeta.Name `-inference-pool` }}",
                                "--pool-namespace",
                                "{{ .ObjectMeta.Namespace }}",
                                "--pool-group",
                                "inference.networking.x-k8s.io",
                                "--zap-encoder",
                                "json",
                                "--grpc-port",
                                "9002",
                                "--grpc-health-port",
                                "9003",
                                "--secure-serving",
                                "--model-server-metrics-scheme",
                                "https",
                                "--cert-path",
                                "/var/run/kserve/tls",
                                "--config-text",
                                yaml.dump(ROUTER_SCHEDULER_CONFIG_ESTIMATED_PREFIX_CACHE),
                            ],
                        }
                    ],
                }
            },
            "route": {},
            "gateway": {},
        },
        disable_scheduler=False,
        enable_prefill_decode=False,
        wait=True,
        timeout=Timeout.TIMEOUT_15MIN,
    ) as llm_service:
        yield llm_service


@pytest.fixture(scope="class")
def singlenode_precise_prefix_cache(
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    llmd_s3_service_account: ServiceAccount,
    llmd_gateway: Gateway,
) -> Generator[LLMInferenceService, None, None]:
    """LLMInferenceService fixture for single-node precise prefix cache test."""

    llmisvc_name = "singlenode-precise-prefix-cache"
    kv_transfer_config = {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
    # We use $(POD_IP) and $(MODEL_NAME) for Kubernetes-native variable expansion
    kv_events_config = {
        "enable_kv_cache_events": True,
        "publisher": "zmq",
        "endpoint": f"tcp://{llmisvc_name}-epp-service:5557",
        "topic": "kv@$(POD_IP)@$(MODEL_NAME)",
    }

    with create_llmisvc(
        client=admin_client,
        name=llmisvc_name,
        namespace=unprivileged_model_namespace.name,
        storage_uri=ModelStorage.HF_QWEN_7B_INSTRUCT,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        replicas=2,
        annotations={
            "prometheus.io/port": "8000",
            "prometheus.io/path": "/metrics",
        },
        container_resources={
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
        container_env=[
            # Define dependencies FIRST so they can be expanded later
            {
                "name": "POD_IP",
                "valueFrom": {"fieldRef": {"apiVersion": "v1", "fieldPath": "status.podIP"}},
            },
            {"name": "MODEL_NAME", "value": "Qwen/Qwen2.5-7B-Instruct"},
            # Application Settings
            {"name": "VLLM_LOGGING_LEVEL", "value": "DEBUG"},
            {"name": "PYTHONHASHSEED", "value": str(PREFIX_CACHE_HASH_SEED)},
            # vLLM Args
            {
                "name": "VLLM_ADDITIONAL_ARGS",
                "value": (
                    f"--enable-prefix-caching "
                    f"--block-size {PREFIX_CACHE_BLOCK_SIZE} "
                    f"--kv_transfer_config '{json.dumps(kv_transfer_config)}' "
                    f"--kv-events-config '{json.dumps(kv_events_config)}'"
                ),
            },
        ],
        liveness_probe=LLMD_LIVENESS_PROBE,
        service_account=llmd_s3_service_account.name,
        enable_auth=True,
        router_config={
            "scheduler": {
                "template": {
                    "volumes": [{"name": "tokenizers", "emptyDir": {}}],
                    "containers": [
                        {
                            "name": "main",
                            "ports": [
                                {"name": "zmq-kv", "containerPort": 5557, "protocol": "TCP"},
                                {"name": "grpc-rpc", "containerPort": 9002, "protocol": "TCP"},
                                {"name": "grpc-health", "containerPort": 9003, "protocol": "TCP"},
                                {"name": "metrics", "containerPort": 9090, "protocol": "TCP"},
                            ],
                            "env": [
                                {"name": "HF_HOME", "value": "/mnt/tokenizers"},
                            ],
                            "volumeMounts": [
                                {
                                    "name": "tokenizers",
                                    "mountPath": "/mnt/tokenizers",
                                    "readOnly": False,
                                }
                            ],
                            "args": [
                                "--v=4",
                                "--zap-encoder",
                                "json",
                                "--pool-name",
                                "{{ ChildName .ObjectMeta.Name `-inference-pool` }}",
                                "--pool-namespace",
                                "{{ .ObjectMeta.Namespace }}",
                                "--pool-group",
                                "inference.networking.x-k8s.io",
                                "--grpc-port",
                                "9002",
                                "--grpc-health-port",
                                "9003",
                                "--secure-serving",
                                "--model-server-metrics-scheme",
                                "https",
                                "--model-server-metrics-https-insecure-skip-verify",
                                "--cert-path",
                                "/var/run/kserve/tls",
                                "--config-text",
                                yaml.dump(ROUTER_SCHEDULER_CONFIG_PRECISE_PREFIX_CACHE),
                            ],
                        }
                    ],
                }
            },
            "route": {},
            "gateway": {},
        },
        disable_scheduler=False,
        enable_prefill_decode=False,
        wait=True,
        timeout=Timeout.TIMEOUT_15MIN,
    ) as llm_service:
        yield llm_service
