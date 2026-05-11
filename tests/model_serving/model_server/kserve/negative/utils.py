"""Utility functions for negative inference tests."""

import shlex
from typing import Any

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from pyhelper_utils.shell import run_command
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_serving.model_server.kserve.negative.constants import KSERVE_CONTROL_PLANE_DEPLOYMENTS
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


def wait_for_isvc_ready_false(
    isvc: InferenceService,
    *,
    timeout: int = Timeout.TIMEOUT_15MIN,
    sleep: int = 5,
) -> None:
    """Poll until ISVC ``status.conditions`` contains ``Ready=False``.

    Args:
        isvc: InferenceService to observe.
        timeout: Maximum seconds to wait.
        sleep: Seconds between polls.

    Raises:
        TimeoutExpiredError: If the condition is not observed within ``timeout``.
    """

    def _ready_condition() -> Any:
        isvc.update()
        inst_status = getattr(isvc.instance, "status", None)
        if not inst_status:
            return None
        conditions = getattr(inst_status, "conditions", None)
        if not conditions:
            return None
        for cond in conditions:
            if cond.type == "Ready":
                return cond
        return None

    sample: Any = None
    try:
        for sample in TimeoutSampler(wait_timeout=timeout, sleep=sleep, func=_ready_condition):
            if sample is not None and getattr(sample, "status", None) == "False":
                LOGGER.info(
                    "InferenceService Ready=False observed",
                    isvc=isvc.name,
                    namespace=isvc.namespace,
                    reason=getattr(sample, "reason", None),
                    message=getattr(sample, "message", None),
                )
                return
    except TimeoutExpiredError:
        LOGGER.error(
            "Timed out waiting for InferenceService Ready=False",
            isvc=isvc.name,
            namespace=isvc.namespace,
            last_ready_condition=sample,
            last_reason=getattr(sample, "reason", None) if sample is not None else None,
            last_message=getattr(sample, "message", None) if sample is not None else None,
        )
        raise


def _pod_restart_total(pod: Pod) -> int:
    total = 0
    for cs in pod.instance.status.containerStatuses or []:
        total += cs.restartCount
    for ics in pod.instance.status.initContainerStatuses or []:
        total += ics.restartCount
    return total


def snapshot_kserve_control_plane_restart_totals(
    admin_client: DynamicClient,
    applications_namespace: str,
    deployment_names: tuple[str, ...] = KSERVE_CONTROL_PLANE_DEPLOYMENTS,
) -> dict[str, int]:
    """Sum container restart counts for each KServe / ODH model controller deployment."""
    totals: dict[str, int] = {}
    for name in deployment_names:
        dep = Deployment(client=admin_client, name=name, namespace=applications_namespace)
        if not dep.exists:
            raise AssertionError(f"Deployment {name!r} not found in namespace {applications_namespace!r}")
        match = dep.instance.spec.selector.matchLabels
        if not match:
            raise AssertionError(f"Deployment {name!r} has empty pod selector in {applications_namespace!r}")
        label_selector = ",".join(f"{k}={v}" for k, v in sorted(match.items()))
        pods = list(Pod.get(client=admin_client, namespace=applications_namespace, label_selector=label_selector))
        totals[name] = sum(_pod_restart_total(p) for p in pods)
    return totals


def assert_kserve_control_plane_stable(
    admin_client: DynamicClient,
    applications_namespace: str,
    prior_restart_totals: dict[str, int],
    deployment_names: tuple[str, ...] = KSERVE_CONTROL_PLANE_DEPLOYMENTS,
) -> None:
    """Assert control-plane deployments stayed Available without new crashes or restarts.

    Args:
        admin_client: Cluster admin API client.
        applications_namespace: Namespace where KServe and ODH model controllers run.
        prior_restart_totals: Snapshot from ``snapshot_kserve_control_plane_restart_totals`` taken
            before the scenario under test.
        deployment_names: Deployments to verify (defaults to KServe + ODH model controller).

    Raises:
        AssertionError: If a deployment is missing, not Available, has CrashLoopBackOff, or
            container restart totals increased compared to ``prior_restart_totals``.
    """
    for name in deployment_names:
        dep = Deployment(client=admin_client, name=name, namespace=applications_namespace)
        assert dep.exists, f"Deployment {name!r} missing in {applications_namespace!r}"

        status = getattr(dep.instance, "status", None)
        conditions = getattr(status, "conditions", None) or []
        available = any(c.type == "Available" and c.status == "True" for c in conditions)
        assert available, (
            f"Deployment {name!r} in {applications_namespace!r} is not Available after negative ISVC scenario"
        )

        match = dep.instance.spec.selector.matchLabels
        assert match, f"Deployment {name!r} has empty pod selector"
        label_selector = ",".join(f"{k}={v}" for k, v in sorted(match.items()))
        pods = list(Pod.get(client=admin_client, namespace=applications_namespace, label_selector=label_selector))
        assert pods, f"No pods found for deployment {name!r} in {applications_namespace!r}"

        for pod in pods:
            phase = pod.instance.status.phase
            assert phase != "Failed", (
                f"Pod {pod.name!r} for deployment {name!r} is Failed after negative ISVC scenario (phase={phase!r})"
            )
            for cs in pod.instance.status.containerStatuses or []:
                waiting = getattr(getattr(cs, "state", None), "waiting", None)
                if waiting is not None and waiting.reason == "CrashLoopBackOff":
                    raise AssertionError(f"Container {cs.name!r} in pod {pod.name!r} ({name!r}) is in CrashLoopBackOff")
            for ics in pod.instance.status.initContainerStatuses or []:
                waiting = getattr(getattr(ics, "state", None), "waiting", None)
                if waiting is not None and waiting.reason == "CrashLoopBackOff":
                    raise AssertionError(
                        f"Init container {ics.name!r} in pod {pod.name!r} ({name!r}) is in CrashLoopBackOff"
                    )

        current_total = sum(_pod_restart_total(p) for p in pods)
        before = prior_restart_totals[name]
        assert current_total == before, (
            f"Deployment {name!r} container restart total changed ({before} -> {current_total}); "
            "control plane may have restarted due to the invalid S3 InferenceService"
        )


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
