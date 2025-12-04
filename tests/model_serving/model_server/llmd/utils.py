"""
Utility functions for LLM Deployment (LLMD) tests.

This module provides helper functions for LLMD test operations using ocp_resources.
Follows the established model server utils pattern for consistency.
"""

import re
import time
from typing import Any

from kubernetes.dynamic import DynamicClient
from ocp_resources.gateway import Gateway
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler

from utilities.constants import Protocols
from utilities.exceptions import PodContainersRestartError
from utilities.llmd_utils import verify_inference_response_llmd
from utilities.manifests.tinyllama import TINYLLAMA_INFERENCE_CONFIG


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
        dyn_client=client,
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
        dyn_client=client,
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


def count_chat_completions_requests_in_pod(pod: Pod) -> int:
    """
    Count POST /v1/chat/completions requests in pod logs.

    Args:
        pod: The vLLM workload pod to check

    Returns:
        Number of successful chat completion requests found in logs
    """
    try:
        logs = pod.log(container="main", since_seconds=120)

        # Match: "POST /v1/chat/completions HTTP/1.1" 200
        pattern = r"POST /v1/chat/completions HTTP/1.1.*200"
        matches = re.findall(pattern, logs)

        LOGGER.info(f"Pod {pod.name}: Found {len(matches)} requests matching pattern")

        # Debug: Show sample log lines if no matches found
        if len(matches) == 0:
            log_lines = logs.split("\n")
            LOGGER.info(f"Pod {pod.name}: Total log lines: {len(log_lines)}")
            # Show lines containing "POST" or "completions"
            relevant_lines = [line for line in log_lines if "POST" in line or "completion" in line.lower()]
            if relevant_lines:
                LOGGER.info(f"Pod {pod.name}: Sample relevant lines (first 5):")
                for line in relevant_lines[:5]:
                    LOGGER.info(f"  {line[:200]}")

        return len(matches)
    except Exception as e:
        LOGGER.info(f"Failed to count requests for pod {pod.name}: {e}")
        return 0


def get_pod_that_handled_request(workload_pods: list[Pod], baseline_counts: dict[str, int]) -> str | None:
    """
    Determine which pod handled a request by counting POST requests.

    Args:
        workload_pods: List of vLLM workload pods
        baseline_counts: Dict of {pod_name: request_count} before this request

    Returns:
        Pod name that handled the request, or None if not found
    """
    current_counts = {}
    for pod in workload_pods:
        current_counts[pod.name] = count_chat_completions_requests_in_pod(pod=pod)

    for pod in workload_pods:
        baseline = baseline_counts.get(pod.name, 0)
        current = current_counts.get(pod.name, 0)

        if current > baseline:
            LOGGER.info(f"Pod {pod.name} handled request: {baseline} -> {current} (+{current - baseline})")
            return pod.name

    LOGGER.warning("Could not determine which pod handled request")
    return None


def verify_singlenode_prefix_cache_routing(
    llmisvc: LLMInferenceService,
    token: str,
    workload_pods: list[Pod],
) -> None:
    """
    Test single-node precise prefix cache routing with block-level caching validation.

    This function validates the core concept of precise prefix caching:
    - Repeated prompts should hit the same pod (full cache)
    - Prompts with shared prefixes should route to same pod (partial cache)
    - Different prompts may distribute across pods (load balancing)

    Args:
        llmisvc: The LLMInferenceService to send requests to
        token: Authentication token
        workload_pods: List of vLLM workload pods to check routing
    """
    LOGGER.info("Testing precise prefix cache routing")

    # Initialize baseline request counts
    baseline_counts = {}

    for pod in workload_pods:
        baseline_counts[pod.name] = count_chat_completions_requests_in_pod(pod=pod)

    # Phase 1: Repeated prompts (full cache hit)
    LOGGER.info("Phase 1: Testing repeated prompts")
    repeated_prompt = (
        "Explain in detail the fundamental principles of quantum mechanics including "
        "wave-particle duality, superposition, and entanglement in simple terms. "
        "Additionally, describe how these quantum phenomena differ from classical physics "
        "and why they are important for understanding the nature of reality at the atomic scale."
    )

    phase1_pods = []
    for i in range(3):
        inference_config = {
            "default_query_model": {
                "query_input": repeated_prompt,
                "query_output": r".*",
                "use_regex": True,
            },
            "chat_completions": TINYLLAMA_INFERENCE_CONFIG["chat_completions"],
        }

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

        handling_pod = get_pod_that_handled_request(
            workload_pods=workload_pods,
            baseline_counts=baseline_counts,
        )
        phase1_pods.append(handling_pod)
        if handling_pod:
            baseline_counts[handling_pod] = baseline_counts.get(handling_pod, 0) + 1

    # Verify routing affinity for repeated prompts
    unique_phase1_pods = set(p for p in phase1_pods if p is not None)
    assert len(unique_phase1_pods) == 1, f"Repeated prompts should route to same pod, got {unique_phase1_pods}"
    LOGGER.info(f"Phase 1: All repeated requests routed to {unique_phase1_pods}")

    # Phase 2: Shared prefix prompts (partial cache hit)
    LOGGER.info("Phase 2: Testing shared prefix prompts")
    prefix = (
        "Explain in detail the fundamental principles of quantum mechanics including "
        "wave-particle duality, the concept of superposition, the measurement problem, "
        "the Copenhagen interpretation, and how these principles challenge our classical "
        "understanding of reality and determinism in"
    )
    shared_prefix_prompts = [
        f"{prefix} modern physics",
        f"{prefix} quantum computing applications",
        f"{prefix} the study of subatomic particles",
    ]

    phase2_pods = []
    for prompt in shared_prefix_prompts:
        inference_config = {
            "default_query_model": {
                "query_input": prompt,
                "query_output": r".*",
                "use_regex": True,
            },
            "chat_completions": TINYLLAMA_INFERENCE_CONFIG["chat_completions"],
        }

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

        handling_pod = get_pod_that_handled_request(
            workload_pods=workload_pods,
            baseline_counts=baseline_counts,
        )
        phase2_pods.append(handling_pod)
        if handling_pod:
            baseline_counts[handling_pod] = baseline_counts.get(handling_pod, 0) + 1

    # Verify routing affinity for shared prefix prompts
    unique_phase2_pods = set(p for p in phase2_pods if p is not None)
    assert len(unique_phase2_pods) == 1, f"Shared prefix prompts should route to same pod, got {unique_phase2_pods}"
    LOGGER.info(f"Phase 2: All shared prefix requests routed to {unique_phase2_pods}")

    LOGGER.info("All cache routing tests completed successfully")
