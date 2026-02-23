"""
Utility functions for LLM Deployment (LLMD) tests.

This module provides helper functions for LLMD test operations using ocp_resources.
Follows the established model server utils pattern for consistency.
"""

from typing import Any

from kubernetes.dynamic import DynamicClient
from ocp_resources.gateway import Gateway
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.pod import Pod
from ocp_resources.prometheus import Prometheus
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler, retry

from tests.model_serving.model_server.llmd.constants import PREFIX_CACHE_BLOCK_SIZE
from utilities.constants import Protocols
from utilities.exceptions import PodContainersRestartError
from utilities.llmd_utils import verify_inference_response_llmd
from utilities.manifests.tinyllama import TINYLLAMA_INFERENCE_CONFIG
from utilities.monitoring import get_metrics_value

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
    from ocp_resources.resource import Resource

    from utilities.exceptions import FailedPodsError

    LOGGER.info(f"Comprehensive health check for LLMInferenceService {llm_service.name}")

    container_wait_base_errors = ["InvalidImageName", "CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull"]
    container_terminated_base_errors = [Resource.Status.ERROR, "CrashLoopBackOff"]

    def get_llmd_pods():
        """Get LLMD workload pods for this LLMInferenceService."""
        pods = []
        for pod in Pod.get(
            client=client,
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


def get_llmd_workload_pods(
    client: DynamicClient,
    llmisvc: LLMInferenceService,
) -> list[Pod]:
    """
    Get all workload pods for an LLMInferenceService.

    Args:
        client: DynamicClient instance
        llmisvc: The LLMInferenceService to get pods for

    Returns:
        List of workload Pod objects
    """
    pods = []
    for pod in Pod.get(
        client=client,
        namespace=llmisvc.namespace,
        label_selector=(
            f"{Pod.ApiGroup.APP_KUBERNETES_IO}/part-of=llminferenceservice,"
            f"{Pod.ApiGroup.APP_KUBERNETES_IO}/name={llmisvc.name}"
        ),
    ):
        labels = pod.instance.metadata.get("labels", {})
        if labels.get("kserve.io/component") == "workload":
            pods.append(pod)
    return pods


def get_llmd_router_scheduler_pod(
    client: DynamicClient,
    llmisvc: LLMInferenceService,
) -> Pod | None:
    """
    Get the router-scheduler pod for an LLMInferenceService.

    Args:
        client: DynamicClient instance
        llmisvc: The LLMInferenceService to get router-scheduler pod for

    Returns:
        Router-scheduler Pod object or None if not found
    """
    for pod in Pod.get(
        client=client,
        namespace=llmisvc.namespace,
        label_selector=(
            f"{Pod.ApiGroup.APP_KUBERNETES_IO}/part-of=llminferenceservice,"
            f"{Pod.ApiGroup.APP_KUBERNETES_IO}/name={llmisvc.name}"
        ),
    ):
        labels = pod.instance.metadata.get("labels", {})
        if labels.get(f"{Pod.ApiGroup.APP_KUBERNETES_IO}/component") == "llminferenceservice-router-scheduler":
            return pod
    return None


def send_prefix_cache_test_requests(
    llmisvc: LLMInferenceService,
    token: str,
    num_requests: int = 20,
) -> int:
    """
    Send N identical requests to validate prefix cache.

    This function sends the same prompt multiple times to test cache affinity.
    All requests after the first should hit the cache and route to the same pod.

    Args:
        llmisvc: The LLMInferenceService to send requests to
        token: Authentication token
        num_requests: Number of identical requests to send (default 20)

    Returns:
        int: Number of successful requests completed
    """
    successful_requests = 0
    failed_requests = 0

    # Single prompt to be cached
    cached_prompt = (
        "Explain in detail the fundamental principles of quantum mechanics including "
        "wave-particle duality, superposition, and entanglement in simple terms. "
        "Additionally, describe how these quantum phenomena differ from classical physics "
        "and why they are important for understanding the nature of reality at the atomic scale."
    )

    LOGGER.info(f"Sending {num_requests} identical requests to test prefix cache")

    for index in range(num_requests):
        LOGGER.info(f"Sending request {index + 1}/{num_requests}")
        inference_config = {
            "default_query_model": {
                "query_input": cached_prompt,
                "query_output": r".*",
                "use_regex": True,
            },
            "chat_completions": TINYLLAMA_INFERENCE_CONFIG["chat_completions"],
        }

        try:
            verify_inference_response_llmd(
                llm_service=llmisvc,
                inference_config=inference_config,
                inference_type="chat_completions",
                protocol=Protocols.HTTPS,
                use_default_query=True,
                insecure=False,
                model_name=llmisvc.instance.spec.model.name,
                token=token,
                authorized_user=True,
            )
            successful_requests += 1
        except Exception as e:  # noqa: BLE001
            LOGGER.error(f"Request {index + 1} failed: {e}")
            failed_requests += 1

    # Log statistics
    LOGGER.info(f"{successful_requests}/{num_requests} requests completed successfully")

    return successful_requests


def get_successful_requests_by_pod(
    prometheus: Prometheus,
    llmisvc: LLMInferenceService,
    pods: list[Pod],
) -> dict[str, float]:
    """
    Retrieves the total number of successful requests per pod.

    This function queries the `kserve_vllm:request_success_total` counter metric
    from Prometheus for the specified inference service.

    Args:
        prometheus: The Prometheus client instance.
        llmisvc: The LLM Inference Service object to filter by.
        pods: A list of pod names to include in the result.

    Returns:
        dict[str, float]: A dictionary mapping pod names to their respective
            total successful request counts.
    """
    success_counts: dict[str, float] = {}

    for pod in pods:
        query = f'sum(kserve_vllm:request_success_total{{namespace="{llmisvc.namespace}",pod="{pod.name}"}})'
        count = float(get_metrics_value(prometheus=prometheus, metrics_query=query) or 0)
        success_counts[pod.name] = count

    return success_counts


def get_prefix_cache_hits_by_pod(
    prometheus: Prometheus,
    llmisvc: LLMInferenceService,
    pods: list[Pod],
) -> dict[str, float]:
    """
    Retrieves the total number of prefix cache hits per pod.

    This function queries the `kserve_vllm:prefix_cache_hits_total` counter metric
    from Prometheus for the specified inference service.

    Args:
        prometheus: The Prometheus client instance.
        llmisvc: The LLM Inference Service object to filter by.
        pods: A list of pod names to include in the result.

    Returns:
        dict[str, float]: A dictionary mapping pod names to their respective
            total prefix cache hit counts.
    """
    cache_hits: dict[str, float] = {}

    for pod in pods:
        query = f'sum(kserve_vllm:prefix_cache_hits_total{{namespace="{llmisvc.namespace}",pod="{pod.name}"}})'
        count = float(get_metrics_value(prometheus=prometheus, metrics_query=query) or 0)
        cache_hits[pod.name] = count

    return cache_hits


@retry(wait_timeout=90, sleep=30, exceptions_dict={AssertionError: []}, print_log=False)
def verify_estimated_prefix_cache(
    prometheus: Prometheus,
    llmisvc: LLMInferenceService,
    workload_pods: list[Pod],
    expected_requests: int,
) -> bool:
    """
    Verify that the Estimated Prefix Cache is working correctly via metric assertions.

    This function polls Prometheus to assess two key behaviors:
    1. all traffic was routed to a single pod
    2. the number of prefix cache hits matches

    Retries for up to 90s to allow for metric scraping latency.

    Args:
        prometheus: Prometheus client.
        llmisvc: Target Inference Service.
        workload_pods: List of serving pods.
        expected_requests: Total expected request count.

    Returns:
        bool: True if verification succeeds (required by @retry decorator).

    Raises:
        AssertionError: If validation fails after the retry timeout.
    """
    LOGGER.info("Checking Estimated Prefix Cache logic...")

    # 1. Verify all traffic is routed to a single pod
    request_counts = get_successful_requests_by_pod(
        prometheus=prometheus,
        llmisvc=llmisvc,
        pods=workload_pods,
    )
    LOGGER.info(f"Request count by pod: {request_counts}")

    # All requests must be routed to exactly one pod (prefix cache affinity).
    # This assertion works regardless of the number of pods in the deployment.
    pods_with_traffic = [pod for pod, count in request_counts.items() if count > 0]
    assert len(pods_with_traffic) == 1, (
        f"Expected all traffic to be routed to exactly 1 pod, but {len(pods_with_traffic)} pods received traffic. "
        f"Distribution: {request_counts}"
    )

    active_pod = pods_with_traffic[0]
    assert request_counts[active_pod] == expected_requests, (
        f"Expected {expected_requests} requests on the active pod '{active_pod}', but got {request_counts[active_pod]}"
    )

    # 2. Verify Prefix Cache Hits on the active pod
    # The first request warms the cache, subsequent requests should hit it.
    cache_hit_counts = get_prefix_cache_hits_by_pod(
        prometheus=prometheus,
        llmisvc=llmisvc,
        pods=workload_pods,
    )
    LOGGER.info(f"Prefix cache hits by pod: {cache_hit_counts}")

    # Logic: (N-1) requests * Block Size
    expected_hits = (expected_requests - 1) * PREFIX_CACHE_BLOCK_SIZE

    assert cache_hit_counts[active_pod] == expected_hits, (
        f"Cache hit mismatch on active pod '{active_pod}'. "
        f"Expected {expected_hits} hits, got {cache_hit_counts[active_pod]}"
    )

    return True
