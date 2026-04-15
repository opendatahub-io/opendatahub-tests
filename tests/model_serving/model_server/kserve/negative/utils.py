"""Utility functions for negative inference tests."""

import shlex
from typing import Any

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from pyhelper_utils.shell import run_command
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from utilities.constants import Timeout
from utilities.infra import get_pods_by_isvc_label
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG

LOGGER = structlog.get_logger(name=__name__)

VALID_OVMS_INFERENCE_BODY: dict[str, Any] = {
    "inputs": ONNX_INFERENCE_CONFIG["default_query_model"]["infer"]["query_input"]
}


def assert_pods_healthy(
    admin_client: DynamicClient,
    isvc: InferenceService,
    initial_pod_state: dict[str, dict[str, Any]],
) -> None:
    """Assert that all pods remain running with no restarts compared to initial state.

    Args:
        admin_client: Kubernetes client with admin privileges.
        isvc: The InferenceService whose pods to check.
        initial_pod_state: Mapping of pod UIDs to their initial state
            (name, restart counts) captured before the test action.
    """
    current_pods = get_pods_by_isvc_label(client=admin_client, isvc=isvc)
    assert len(current_pods) > 0, "No pods found for the InferenceService"

    current_pod_uids = {pod.instance.metadata.uid for pod in current_pods}
    initial_pod_uids = set(initial_pod_state.keys())
    assert current_pod_uids == initial_pod_uids, (
        f"Pod UIDs changed after invalid requests. "
        f"Initial: {initial_pod_uids}, Current: {current_pod_uids}. "
        f"This indicates pods were recreated."
    )

    for pod in current_pods:
        uid = pod.instance.metadata.uid
        initial_state = initial_pod_state[uid]
        assert pod.instance.status.phase == "Running", (
            f"Pod {pod.name} is not running, status: {pod.instance.status.phase}"
        )
        for container in pod.instance.status.containerStatuses or []:
            initial_restart_count = initial_state["restart_counts"].get(container.name, 0)
            assert container.restartCount == initial_restart_count, (
                f"Container {container.name} in pod {pod.name} restarted. "
                f"Initial: {initial_restart_count}, Current: {container.restartCount}"
            )


def wait_for_isvc_model_status_states(
    isvc: InferenceService,
    *,
    target_model_state: str,
    transition_status: str,
    timeout: int = Timeout.TIMEOUT_15MIN,
    sleep: int = 5,
) -> None:
    """Poll until ``status.modelStatus`` matches the expected model and transition states.

    Args:
        isvc: InferenceService to observe (must already exist on the API server).
        target_model_state: Expected ``states.targetModelState`` (e.g. ``FailedToLoad``).
        transition_status: Expected ``transitionStatus`` (e.g. ``BlockedByFailedLoad``).
        timeout: Maximum seconds to wait.
        sleep: Seconds between polls.

    Raises:
        TimeoutExpiredError: If the status is not observed within ``timeout``.
    """

    def _model_status() -> Any:
        inst_status = getattr(isvc.instance, "status", None)
        if not inst_status:
            return None
        return getattr(inst_status, "modelStatus", None)

    sample: Any = None
    try:
        for sample in TimeoutSampler(wait_timeout=timeout, sleep=sleep, func=_model_status):
            if not sample or not getattr(sample, "states", None):
                continue
            states = sample.states
            if states.targetModelState == target_model_state and sample.transitionStatus == transition_status:
                LOGGER.info(
                    "InferenceService model status reached expected state",
                    isvc=isvc.name,
                    namespace=isvc.namespace,
                    target_model_state=target_model_state,
                    transition_status=transition_status,
                )
                return
    except TimeoutExpiredError:
        LOGGER.error(
            "Timed out waiting for InferenceService model status",
            isvc=isvc.name,
            namespace=isvc.namespace,
            last_model_status=sample,
        )
        raise


def send_inference_request(
    inference_service: InferenceService,
    body: str,
    model_name: str | None = None,
    content_type: str = "application/json",
) -> tuple[int, str]:
    """Send an inference request and return HTTP status code and response body.

    Unlike UserInference, this function does not retry or raise on error
    status codes, making it suitable for negative testing where error
    responses are the expected outcome.

    Args:
        inference_service: The InferenceService to send the request to.
        body: The raw string payload (can be invalid JSON for negative testing).
        model_name: Override the model name in the URL path.
            Defaults to the InferenceService name.
        content_type: The Content-Type header value. Defaults to "application/json".

    Returns:
        A tuple of (status_code, response_body).

    Raises:
        ValueError: If the InferenceService has no URL or curl output is malformed.
    """
    base_url = inference_service.instance.status.url
    if not base_url:
        raise ValueError(f"InferenceService '{inference_service.name}' has no URL; is it Ready?")

    target_model = model_name or inference_service.name
    endpoint = f"{base_url}/v2/models/{target_model}/infer"

    cmd = (
        f"curl -s -w '\\n%{{http_code}}' "
        f"-X POST {endpoint} "
        f"-H 'Content-Type: {content_type}' "
        f"--data-raw {shlex.quote(body)} "
        f"--insecure"
    )

    _, out, _ = run_command(command=shlex.split(cmd), verify_stderr=False, check=False)

    lines = out.strip().split("\n")
    try:
        status_code = int(lines[-1])
    except ValueError as exc:
        raise ValueError(f"Could not parse HTTP status code from curl output: {out!r}") from exc
    return status_code, "\n".join(lines[:-1])
