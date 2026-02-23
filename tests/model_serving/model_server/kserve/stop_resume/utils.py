"""Utilities for stop/resume model testing."""

import time

from kubernetes.dynamic.client import DynamicClient
from ocp_resources.inference_service import InferenceService
from timeout_sampler import TimeoutExpiredError

from tests.model_serving.model_server.utils import verify_no_inference_pods


def consistently_verify_no_pods_exist(
    client: DynamicClient,
    isvc: InferenceService,
    checks: int = 10,
    interval: int = 1,
) -> bool:
    """
    Verify that no inference pods exist for the given inference service.

    Args:
        client: The Kubernetes client
        isvc: The InferenceService object
        checks: Number of checks to perform (default: 10)

    Returns:
        bool: True if no pods exist (verification passed), False if pods are found
    """
    try:
        for _ in range(checks):
            if not verify_no_inference_pods(client, isvc):
                return False
            # Nested timeout samplers can cause false negatives if the internal sampler has
            # a timeout that is greater than the external sampler.
            # So we iterate and sleep here instead.
            time.sleep(seconds=interval)
    except TimeoutExpiredError:
        return False
    return True
