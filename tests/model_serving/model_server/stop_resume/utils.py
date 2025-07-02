"""Utilities for stop/resume model testing."""

from kubernetes.dynamic.client import DynamicClient
from ocp_resources.inference_service import InferenceService

from timeout_sampler import TimeoutSampler, TimeoutExpiredError
from tests.model_serving.model_server.serverless.utils import verify_no_inference_pods


def verify_no_pods_exist_with_timeout(
    client: DynamicClient,
    isvc: InferenceService,
    wait_timeout: int = 10,
    sleep: int = 1,
) -> bool:
    """
    Verify that no inference pods exist for the given inference service within a timeout period.

    Args:
        client: The Kubernetes client
        isvc: The InferenceService object
        wait_timeout: Maximum time to wait in seconds (default: 10)
        sleep: Sleep interval between checks in seconds (default: 1)

    Returns:
        bool: True if no pods exist (verification passed), False if pods are found
    """
    try:
        for result in TimeoutSampler(
            wait_timeout=wait_timeout,
            sleep=sleep,
            func=verify_no_inference_pods,
            client=client,
            isvc=isvc,
        ):
            if result is not None:
                return False  # Pods were found when none should exist
    except TimeoutExpiredError:
        # Expected behavior - no pods found during the timeout period
        pass

    return True  # No pods found during the timeout period
