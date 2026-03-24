"""Utilities for inference service lifecycle testing."""

import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_serving.model_server.utils import verify_no_inference_pods
from utilities.constants import Timeout
from utilities.infra import get_pods_by_isvc_label
from utilities.opendatahub_logger import get_logger

LOGGER = get_logger(name=__name__)


@contextmanager
def update_inference_service(
    client: DynamicClient, isvc: InferenceService, isvc_updated_dict: dict[str, Any], wait_for_new_pods: bool = True
) -> Generator[InferenceService, Any]:
    """
    Update InferenceService object.

    Args:
        client (DynamicClient): DynamicClient object.
        isvc (InferenceService): InferenceService object.
        isvc_updated_dict (dict[str, Any]): Dictionary of InferenceService fields to update.
        wait_for_new_pods (bool): Whether to wait for new pods to be created.

    """
    orig_pods = get_pods_by_isvc_label(client=client, isvc=isvc)

    with ResourceEditor(patches={isvc: isvc_updated_dict}):
        if wait_for_new_pods:
            wait_for_new_running_inference_pods(isvc=isvc, orig_pods=orig_pods)

        yield isvc


def verify_env_vars_in_isvc_pods(isvc: InferenceService, env_vars: list[dict[str, str]], vars_exist: bool) -> None:
    """
    Verify that the environment variables in the InferenceService pods match the expected values.

    Args:
        isvc (InferenceService): InferenceService object.
        env_vars (list[dict[str, str]]): List of environment variables to verify.
        vars_exist (bool): Whether the environment variables should exist in the pod.

    Raises:
        ValueError: If the environment variables do not match the expected values.
    """
    unset_pods = []
    checked_env_vars_names = [env_var["name"] for env_var in env_vars]

    pods = get_pods_by_isvc_label(client=isvc.client, isvc=isvc)

    for pod in pods:
        pod_env_vars_names = [env_var.name for env_var in pod.instance.spec.containers[0].get("env", [])]
        envs_in_pod = [env_var in pod_env_vars_names for env_var in checked_env_vars_names]

        if vars_exist:
            if not all(envs_in_pod):
                unset_pods.append(pod.name)

        else:
            if all(envs_in_pod):
                unset_pods.append(pod.name)

    if unset_pods:
        raise ValueError(
            f"The environment variables are {'not' if vars_exist else ''} set in the following pods: {unset_pods}"
        )


def wait_for_new_running_inference_pods(
    isvc: InferenceService, orig_pods: list[Pod], expected_num_pods: int | None = None
) -> None:
    """
    Wait for the inference pod to be replaced.

    Args:
        isvc (InferenceService): InferenceService object.
        orig_pods (list): List of Pod objects.
        expected_num_pods (int): Number of pods expected to be running. I
            f not provided, the number of pods is expected to be len(orig_pods)

    Raises:
        TimeoutError: If the pods are not replaced.

    """
    LOGGER.info("Waiting for pods to be replaced")
    oring_pods_names = [pod.name for pod in orig_pods]

    expected_num_pods = expected_num_pods or len(orig_pods)

    try:
        for pods in TimeoutSampler(
            wait_timeout=Timeout.TIMEOUT_10MIN,
            sleep=5,
            func=get_pods_by_isvc_label,
            client=isvc.client,
            isvc=isvc,
        ):
            if (
                pods
                and len(pods) == expected_num_pods
                and all(pod.name not in oring_pods_names and pod.status == pod.Status.RUNNING for pod in pods)
            ):
                return

    except TimeoutError:
        LOGGER.error(f"Timeout waiting for pods {oring_pods_names} to be replaced")
        raise


def verify_pull_secret(isvc: InferenceService, pull_secret: str, secret_exists: bool) -> None:
    """
    Verify that the ImagePullSecret in the InferenceService pods match the expected values.

    Args:
        isvc (InferenceService): InferenceService object.
        pull_secret (str): Pull secret to verify
        secret_exists (bool): False if the pull secret should not exist in the pod.

    Raises:
        AssertionError: If the imagePullSecrets do not match the expected presence or name.
    """
    pod = get_pods_by_isvc_label(
        client=isvc.client,
        isvc=isvc,
    )[0]
    image_pull_secrets = pod.instance.spec.imagePullSecrets or []

    secrets = [s.name for s in image_pull_secrets]

    if secret_exists:
        assert secrets, "Expected imagePullSecrets to exist, but none were found."
        assert pull_secret in secrets, f"Expected pull secret '{pull_secret}' not found in imagePullSecrets: {secrets}"
    else:
        assert pull_secret not in secrets, (
            f"Did not expect pull secret '{pull_secret}', but found in imagePullSecrets: {secrets}"
        )


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
            time.sleep(interval)
    except TimeoutExpiredError:
        return False
    return True
