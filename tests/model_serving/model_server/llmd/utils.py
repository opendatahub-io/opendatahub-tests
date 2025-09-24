"""
Utility functions for LLM Deployment (LLMD) tests.

This module provides helper functions for LLMD test operations using ocp_resources.
Follows the established model server utils pattern for consistency.
"""

from kubernetes.dynamic import DynamicClient
from ocp_resources.gateway import Gateway
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger

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


def verify_llmd_pods_not_restarted(client: DynamicClient, llm_service: LLMInferenceService) -> None:
    """
    Verify that LLMD inference pods containers have not restarted.

    This function checks for container restarts in pods related to the specific LLMInferenceService.

    Args:
        client (DynamicClient): DynamicClient instance
        llm_service (LLMInferenceService): The LLMInferenceService to check pods for

    Raises:
        PodContainersRestartError: If any containers in LLMD pods have restarted
    """
    LOGGER.info(f"Verifying that pods for LLMInferenceService {llm_service.name} have not restarted")

    restarted_containers = {}

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
            LOGGER.debug(f"Checking pod {pod.name} for container restarts")

            if pod.instance.status.containerStatuses:
                if _restarted_containers := [
                    container.name for container in pod.instance.status.containerStatuses if container.restartCount > 0
                ]:
                    restarted_containers[pod.name] = _restarted_containers
                    LOGGER.warning(f"Pod {pod.name} has restarted containers: {_restarted_containers}")

    if restarted_containers:
        error_msg = f"LLMD inference containers restarted for {llm_service.name}: {restarted_containers}"
        LOGGER.error(error_msg)
        raise PodContainersRestartError(error_msg)

    LOGGER.info(f"All pods for LLMInferenceService {llm_service.name} have no container restarts")
