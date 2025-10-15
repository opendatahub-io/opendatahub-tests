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
