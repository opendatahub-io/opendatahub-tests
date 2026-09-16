"""Probe utilities for vLLM-Omni initialization and health check validation."""

import time
from typing import Any, Literal

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod

from utilities.constants import Containers, Ports
from utilities.infra import get_pods_by_isvc_label

LOGGER = structlog.get_logger(name=__name__)

ProbeType = Literal["startupProbe", "readinessProbe", "livenessProbe"]

# Timing constants used by health-probe helpers
HEALTH_POLL_INTERVAL_SECONDS: int = 5
POD_APPEAR_TIMEOUT_SECONDS: int = 600


def get_kserve_container(pod: Pod) -> Any:
    """Return the kserve-container spec from the pod spec.

    Args:
        pod: Predictor pod for the vLLM-Omni InferenceService.

    Returns:
        Container spec for kserve-container.

    Raises:
        ValueError: If kserve-container is not found.
    """
    for container in pod.instance.spec.containers:
        if container.name == Containers.KSERVE_CONTAINER_NAME:
            return container
    raise ValueError(f"{Containers.KSERVE_CONTAINER_NAME} not found in pod {pod.name}")


def get_probe(pod: Pod, probe_type: ProbeType) -> dict[str, Any]:
    """Return the requested probe configuration from kserve-container.

    Args:
        pod: Predictor pod for the vLLM-Omni InferenceService.
        probe_type: One of startupProbe, readinessProbe, livenessProbe.

    Returns:
        Probe configuration dict.

    Raises:
        ValueError: If the probe is not configured on the container.
    """
    container = get_kserve_container(pod=pod)
    probe = getattr(container, probe_type, None)
    if not probe:
        raise ValueError(f"{probe_type} not configured on {Containers.KSERVE_CONTAINER_NAME} in pod {pod.name}")
    return dict(probe)


def exec_http_probe(pod: Pod, http_get: dict[str, Any]) -> str:
    """Execute an HTTP GET probe inside the pod and return the HTTP status code.

    Args:
        pod: Predictor pod to exec into.
        http_get: httpGet block from a probe spec (path, port, scheme).

    Returns:
        HTTP status code as a string (e.g. "200"), or "000" on failure.
    """
    path = http_get.get("path", "/health")
    port = http_get.get("port", Ports.REST_PORT)
    scheme = http_get.get("scheme", "HTTP").lower()
    url = f"{scheme}://localhost:{port}{path}"
    curl_cmd = ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "15"]
    if scheme == "https":
        curl_cmd.append("-k")
    curl_cmd.append(url)
    try:
        return pod.execute(command=curl_cmd, container=Containers.KSERVE_CONTAINER_NAME).strip()
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug(event="HTTP probe exec failed; treating as non-200", pod=pod.name, error=str(exc))
        return "000"


def assert_probe_config(
    probe: dict[str, Any],
    probe_name: str,
    expected: dict[str, Any],
) -> None:
    """Assert httpGet config and threshold values match the expected probe spec."""
    http_get = probe.get("httpGet", {})
    assert http_get, f"{probe_name} must define httpGet, got: {probe}"
    assert http_get.get("path") == "/health", (
        f"{probe_name}.httpGet.path must be '/health', got: {http_get.get('path')!r}"
    )
    assert http_get.get("port") == Ports.REST_PORT, (
        f"{probe_name}.httpGet.port must be {Ports.REST_PORT}, got: {http_get.get('port')!r}"
    )
    for key in ("periodSeconds", "failureThreshold", "timeoutSeconds"):
        if key in expected:
            assert probe.get(key) == expected[key], f"{probe_name}.{key} must be {expected[key]}, got: {probe.get(key)}"
    assert "initialDelaySeconds" not in probe, (
        f"{probe_name} must not define initialDelaySeconds (startup probe is the gate)"
    )


def safe_exec_health_probe(pod: Pod, http_get: dict[str, Any]) -> str:
    """Execute /health inside the pod; return '000' on any execution error.

    Returns '000' when the container is still starting and the port is not
    yet open, allowing callers to treat connection failures the same as a
    non-200 HTTP status.

    Args:
        pod: Running predictor pod for the vLLM-Omni InferenceService.
        http_get: httpGet block from a probe spec (path, port, scheme).

    Returns:
        HTTP status code string (e.g. '200', '503') or '000' on exec failure.
    """
    try:
        return exec_http_probe(pod=pod, http_get=http_get)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug(event="health probe exec failed; treating as non-200", pod=pod.name, error=str(exc))
        return "000"


def wait_for_pod_running(admin_client: DynamicClient, isvc: InferenceService) -> Pod:
    """Wait for the first predictor pod to reach Running phase and return it.

    Args:
        admin_client: OpenShift DynamicClient.
        isvc: InferenceService whose predictor pod to wait for.

    Returns:
        First predictor pod in Running phase.

    Raises:
        pytest.fail: If no Running pod appears within POD_APPEAR_TIMEOUT_SECONDS.
    """
    deadline = time.monotonic() + POD_APPEAR_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            pods = get_pods_by_isvc_label(client=admin_client, isvc=isvc)
            for pod in pods:
                if pod.instance.status.phase == "Running":
                    return pod
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug(event="pod not yet Running, retrying", error=str(exc))
        time.sleep(HEALTH_POLL_INTERVAL_SECONDS)

    pytest.fail(f"No Running predictor pod for ISVC '{isvc.name}' appeared within {POD_APPEAR_TIMEOUT_SECONDS} s")


def has_crash_loop_backoff(pod: Pod) -> bool:
    """Return True when any container in the pod is in CrashLoopBackOff state.

    Args:
        pod: Predictor pod to inspect.

    Returns:
        True if at least one container is in CrashLoopBackOff, False otherwise.
    """
    for container_status in pod.instance.status.containerStatuses or []:
        waiting = getattr(getattr(container_status, "state", None), "waiting", None)
        if waiting is not None and waiting.reason == "CrashLoopBackOff":
            return True
    return False
