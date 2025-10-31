"""
Utility functions for LLM Deployment (LLMD) tests.

This module provides helper functions for LLMD test operations using ocp_resources.
Follows the established model server utils pattern for consistency.
"""

from typing import Any, Literal

from kubernetes.dynamic import DynamicClient
from ocp_resources.gateway import Gateway
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler

from utilities.exceptions import PodContainersRestartError


LOGGER = get_logger(name=__name__)


def verify_gateway_status(gateway: Gateway) -> bool:
    """
    Verify that a Gateway is properly configured and programmed.

    Args:
        gateway (Gateway): The Gateway resource to verify

    Returns:
        bool: True if gateway is properly configured, False otherwise
    """
    if not gateway.exists:
        LOGGER.warning(f"Gateway {gateway.name} does not exist")
        return False

    conditions = gateway.instance.status.get("conditions", [])
    for condition in conditions:
        if condition["type"] == "Programmed" and condition["status"] == "True":
            LOGGER.info(f"Gateway {gateway.name} is programmed and ready")
            return True

    LOGGER.warning(f"Gateway {gateway.name} is not in Programmed state")
    return False


def verify_llm_service_status(llm_service: LLMInferenceService) -> bool:
    """
    Verify that an LLMInferenceService is properly configured and ready.

    Args:
        llm_service (LLMInferenceService): The LLMInferenceService resource to verify

    Returns:
        bool: True if service is properly configured, False otherwise
    """
    if not llm_service.exists:
        LOGGER.warning(f"LLMInferenceService {llm_service.name} does not exist")
        return False

    conditions = llm_service.instance.status.get("conditions", [])
    for condition in conditions:
        if condition["type"] == "Ready" and condition["status"] == "True":
            LOGGER.info(f"LLMInferenceService {llm_service.name} is ready")
            return True

    LOGGER.warning(f"LLMInferenceService {llm_service.name} is not in Ready state")
    return False


def verify_llmd_no_failed_pods(
    client: DynamicClient,
    llm_service: LLMInferenceService,
    timeout: int = 300,
) -> None:
    """
    Comprehensive verification that LLMD pods are healthy with no failures.

    This function combines restart detection with comprehensive failure detection,
    similar to verify_no_failed_pods but specifically designed for LLMInferenceService resources.

    Checks for:
    - Container restarts (restartCount > 0)
    - Container waiting states with errors (ImagePullBackOff, CrashLoopBackOff, etc.)
    - Container terminated states with errors
    - Pod failures (CrashLoopBackOff, Failed phases)
    - Pod readiness within timeout

    Args:
        client (DynamicClient): DynamicClient instance
        llm_service (LLMInferenceService): The LLMInferenceService to check pods for
        timeout (int): Timeout in seconds for pod readiness check

    Raises:
        PodContainersRestartError: If any containers have restarted
        FailedPodsError: If any pods are in failed state
        TimeoutError: If pods don't become ready within timeout
    """
    from utilities.exceptions import FailedPodsError
    from ocp_resources.resource import Resource

    LOGGER.info(f"Comprehensive health check for LLMInferenceService {llm_service.name}")

    container_wait_base_errors = ["InvalidImageName", "CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull"]
    container_terminated_base_errors = [Resource.Status.ERROR, "CrashLoopBackOff"]

    def get_llmd_pods():
        """Get LLMD workload pods for this LLMInferenceService."""
        pods = []
        for pod in Pod.get(
            dyn_client=client,
            namespace=llm_service.namespace,
            label_selector=(
                f"{Pod.ApiGroup.APP_KUBERNETES_IO}/part-of=llminferenceservice,"
                f"{Pod.ApiGroup.APP_KUBERNETES_IO}/name={llm_service.name}"
            ),
        ):
            labels = pod.instance.metadata.get("labels", {})
            if labels.get("kserve.io/component") == "workload":
                pods.append(pod)
        return pods

    for pods in TimeoutSampler(
        wait_timeout=timeout,
        sleep=10,
        func=get_llmd_pods,
    ):
        if not pods:
            LOGGER.debug(f"No LLMD workload pods found for {llm_service.name} yet")
            continue

        ready_pods = 0
        failed_pods: dict[str, Any] = {}
        restarted_containers: dict[str, list[str]] = {}
        for pod in pods:
            for condition in pod.instance.status.conditions:
                if condition.type == pod.Status.READY and condition.status == pod.Condition.Status.TRUE:
                    ready_pods += 1
                    break
        if ready_pods == len(pods):
            LOGGER.info(f"All {len(pods)} LLMD pods are ready, performing health checks")

            for pod in pods:
                pod_status = pod.instance.status
                if pod_status.containerStatuses:
                    for container_status in pod_status.get("containerStatuses", []) + pod_status.get(
                        "initContainerStatuses", []
                    ):
                        if hasattr(container_status, "restartCount") and container_status.restartCount > 0:
                            if pod.name not in restarted_containers:
                                restarted_containers[pod.name] = []
                            restarted_containers[pod.name].append(container_status.name)
                            LOGGER.warning(
                                f"Container {container_status.name} in pod {pod.name} has restarted "
                                f"{container_status.restartCount} times"
                            )
                        is_waiting_error = (
                            wait_state := container_status.state.waiting
                        ) and wait_state.reason in container_wait_base_errors

                        is_terminated_error = (
                            terminate_state := container_status.state.terminated
                        ) and terminate_state.reason in container_terminated_base_errors

                        if is_waiting_error or is_terminated_error:
                            failed_pods[pod.name] = pod_status
                            reason = wait_state.reason if is_waiting_error else terminate_state.reason
                            LOGGER.error(
                                f"Container {container_status.name} in pod {pod.name} has error state: {reason}"
                            )
                elif pod_status.phase in (
                    pod.Status.CRASH_LOOPBACK_OFF,
                    pod.Status.FAILED,
                ):
                    failed_pods[pod.name] = pod_status
                    LOGGER.error(f"Pod {pod.name} is in failed phase: {pod_status.phase}")
            if restarted_containers:
                error_msg = f"LLMD containers restarted for {llm_service.name}: {restarted_containers}"
                LOGGER.error(error_msg)
                raise PodContainersRestartError(error_msg)

            if failed_pods:
                LOGGER.error(f"LLMD pods failed for {llm_service.name}: {list(failed_pods.keys())}")
                raise FailedPodsError(pods=failed_pods)

            LOGGER.info(f"All LLMD pods for {llm_service.name} are healthy - no restarts or failures detected")
            return
        LOGGER.debug(f"LLMD pods status: {ready_pods}/{len(pods)} ready for {llm_service.name}")
    raise TimeoutError(f"LLMD pods for {llm_service.name} did not become ready within {timeout} seconds")


def get_kv_transfer_env(vllm_additional_args: str = "") -> list[dict[str, Any]]:
    """Build KV transfer environment variables for prefill-decode.

    Args:
        vllm_additional_args: Additional VLLM arguments to prepend before KV transfer config

    Returns:
        List of environment variable dictionaries for RDMA-based KV cache transfer
    """
    env = [
        {"name": "KSERVE_INFER_ROCE", "value": "true"},
        {"name": "VLLM_NIXL_SIDE_CHANNEL_HOST", "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}}},
        {"name": "UCX_PROTO_INFO", "value": "y"},
        {"name": "UCX_TLS", "value": "rc,sm,self,cuda_copy,cuda_ipc"},
    ]

    # Build KV transfer config
    kv_config = '--kv_transfer_config \'{"kv_connector":"NixlConnector","kv_role":"kv_both"}\''
    if vllm_additional_args:
        vllm_args = f"{vllm_additional_args} {kv_config}"
    else:
        vllm_args = kv_config

    env.append({"name": "VLLM_ADDITIONAL_ARGS", "value": vllm_args})
    return env


def get_scheduler_container_args(scheduler_config_text: str, verbose_level: str = "") -> list[str]:
    """Build scheduler container arguments.

    Args:
        scheduler_config_text: The scheduler configuration YAML text
        verbose_level: Optional verbose level (e.g., "-v=4")

    Returns:
        List of arguments for the scheduler container
    """
    args = []
    if verbose_level:
        args.append(verbose_level)

    args.extend([
        "--pool-name",
        "{{ ChildName .ObjectMeta.Name `-inference-pool` }}",
        "--pool-namespace",
        "{{ .ObjectMeta.Namespace }}",
        "--zap-encoder",
        "json",
        "--grpc-port",
        "9002",
        "--grpc-health-port",
        "9003",
        "--secure-serving",
        "--model-server-metrics-scheme",
        "https",
        "--model-server-metrics-https-insecure-skip-verify",
        "--cert-path",
        "/etc/ssl/certs",
        "--config-text",
        scheduler_config_text,
    ])
    return args


def get_common_multinode_env(
    mode: Literal["rdma", "tcp", "rdma_pd"],
    gpu_memory_util: str = "0.95",
    max_model_len: str = "8192",
    enable_kv_transfer: bool = False,
) -> list[dict[str, Any]]:
    """Build environment variables for multinode configurations.

    Args:
        mode: Network configuration mode
            - "rdma": RDMA-enabled with IBGDa transport
            - "tcp": TCP-only (GCP gVNIC compatible)
            - "rdma_pd": RDMA with prefill-decode separation
        gpu_memory_util: GPU memory utilization (0.0-1.0 as string)
        max_model_len: Maximum sequence length
        enable_kv_transfer: Enable KV cache transfer via NixlConnector (RDMA only)

    Raises:
        ValueError: If enable_kv_transfer=True with mode="tcp"

    Returns:
        List of environment variable dictionaries for the specified mode
    """
    # Validate incompatible parameters
    if enable_kv_transfer and mode == "tcp":
        raise ValueError("KV transfer requires RDMA and cannot be enabled with TCP mode")

    # Base common environment variables
    common_env = [
        {"name": "VLLM_LOGGING_LEVEL", "value": "INFO"},
        {"name": "CUDA_DEVICE_ORDER", "value": "PCI_BUS_ID"},
        {"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "expandable_segments:True"},
        {"name": "NCCL_DEBUG", "value": "WARN"},
        {"name": "NVSHMEM_BOOTSTRAP_TWO_STAGE", "value": "1"},
        {"name": "NVSHMEM_BOOTSTRAP_TIMEOUT", "value": "300"},
        {"name": "NVIDIA_GDRCOPY", "value": "enabled"},
    ]

    # Build VLLM_ADDITIONAL_ARGS
    vllm_args = f"--gpu-memory-utilization {gpu_memory_util} --max-model-len {max_model_len} --enforce-eager"
    if enable_kv_transfer:
        vllm_args += ' --kv_transfer_config \'{"kv_connector":"NixlConnector","kv_role":"kv_both"}\''
    common_env.append({"name": "VLLM_ADDITIONAL_ARGS", "value": vllm_args})

    # Add KV transfer side channel if enabled
    if enable_kv_transfer:
        common_env.append({
            "name": "VLLM_NIXL_SIDE_CHANNEL_HOST",
            "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}},
        })

    # Mode-specific configuration
    if mode in ("rdma", "rdma_pd"):
        # RDMA-specific environment
        rdma_env = [
            {"name": "KSERVE_INFER_ROCE", "value": "true"},
            {"name": "VLLM_ALL2ALL_BACKEND", "value": "deepep_high_throughput"},
            # NCCL configuration for RDMA
            {"name": "NCCL_IB_GID_INDEX", "value": "3"},
            {"name": "NCCL_SOCKET_IFNAME", "value": "net1"},
            {"name": "NCCL_IB_TIMEOUT", "value": "100"},
            # NVSHMEM configuration for RDMA with IBGDa
            {"name": "NVSHMEM_REMOTE_TRANSPORT", "value": "ibgda"},
            {"name": "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME", "value": "net1"},
            {"name": "NVSHMEM_IB_GID_INDEX", "value": "3"},
            {"name": "NVSHMEM_USE_IBGDA", "value": "1"},
            {"name": "NVSHMEM_ENABLE_NIC_PE_MAPPING", "value": "1"},
            {"name": "NVSHMEM_IBGDA_SUPPORT", "value": "1"},
            {"name": "NVSHMEM_IB_ENABLE_IBGDA", "value": "1"},
            {"name": "NVSHMEM_IBGDA_NIC_HANDLER", "value": "gpu"},
            {"name": "NVSHMEM_DEBUG", "value": "WARN"},
            # UCX configuration for RDMA
            {"name": "UCX_TLS", "value": "rc,sm,self,cuda_copy,cuda_ipc"},
            {"name": "UCX_IB_GID_INDEX", "value": "3"},
            {"name": "UCX_RC_MLX5_TM_ENABLE", "value": "n"},
            {"name": "UCX_UD_MLX5_RX_QUEUE_LEN", "value": "1024"},
        ]
        if mode == "rdma_pd":
            rdma_env.append({"name": "UCX_PROTO_INFO", "value": "y"})
        common_env.extend(rdma_env)
    else:  # mode == "tcp"
        # TCP-specific environment
        tcp_env = [
            {"name": "VLLM_API_SERVER_COUNT", "value": "1"},
            {"name": "VLLM_ALL2ALL_BACKEND", "value": "naive"},
            # NCCL configuration for TCP (disable RDMA)
            {"name": "NCCL_IB_DISABLE", "value": "1"},
            {"name": "NCCL_NET_GDR_LEVEL", "value": "0"},
            {"name": "NCCL_P2P_LEVEL", "value": "NVL"},
            {"name": "NCCL_SOCKET_IFNAME", "value": "eth0"},
            {"name": "NCCL_NSOCKS_PERTHREAD", "value": "2"},
            {"name": "NCCL_SOCKET_NTHREADS", "value": "2"},
            {"name": "NCCL_BUFFSIZE", "value": "2097152"},
            # NVSHMEM configuration for TCP with UCX
            {"name": "NVSHMEM_REMOTE_TRANSPORT", "value": "ucx"},
            {"name": "NVSHMEM_DISABLE_CUDA_VMM", "value": "0"},
            {"name": "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME", "value": "eth0"},
            {"name": "NVSHMEM_DEBUG", "value": "INFO"},
            # UCX configuration for TCP
            {"name": "UCX_TLS", "value": "tcp,sm,self,cuda_copy,cuda_ipc"},
            {"name": "UCX_NET_DEVICES", "value": "eth0"},
        ]
        common_env.extend(tcp_env)

    return common_env


def build_scheduler_router_config(
    scheduler_config_text: str,
    verbose_level: str = "",
) -> dict[str, Any]:
    """Build router configuration with scheduler for prefill-decode separation.

    Args:
        scheduler_config_text: YAML configuration for the scheduler
        verbose_level: Optional verbosity level (e.g., "-v=4")

    Returns:
        Router configuration dict with scheduler template
    """
    return {
        "route": {},
        "gateway": {},
        "scheduler": {
            "template": {
                "containers": [
                    {
                        "name": "main",
                        "args": get_scheduler_container_args(
                            scheduler_config_text,
                            verbose_level=verbose_level,
                        ),
                    }
                ]
            }
        },
    }


def build_prefill_config(
    replicas: int,
    env: list[dict[str, Any]],
    resources: dict[str, Any],
    liveness_probe: dict[str, Any],
    parallelism: dict[str, Any] | None = None,
    service_account: str | None = None,
) -> dict[str, Any]:
    """Build prefill configuration for prefill-decode separation.

    Args:
        replicas: Number of prefill replicas
        env: Environment variables for prefill containers
        resources: Container resource limits/requests
        liveness_probe: Liveness probe configuration
        parallelism: Optional parallelism config (for multinode)
        service_account: Optional service account name

    Returns:
        Prefill configuration dict
    """
    prefill_config: dict[str, Any] = {
        "replicas": replicas,
        "template": {
            "containers": [
                {
                    "name": "main",
                    "env": env,
                    "resources": resources,
                    "livenessProbe": liveness_probe,
                }
            ]
        },
    }

    # Add service account if provided
    if service_account:
        prefill_config["template"]["serviceAccountName"] = service_account

    # Add parallelism config if provided (multinode)
    if parallelism:
        prefill_config["parallelism"] = parallelism

        # Add worker config for multinode with parallelism
        prefill_config["worker"] = {
            "containers": [
                {
                    "name": "main",
                    "env": env,
                    "resources": resources,
                }
            ]
        }
        if service_account:
            prefill_config["worker"]["serviceAccountName"] = service_account

    return prefill_config
