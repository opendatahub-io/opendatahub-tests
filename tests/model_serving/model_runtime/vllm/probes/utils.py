"""Utilities for vLLM readiness and liveness probe validation."""

from typing import Any, Literal

from ocp_resources.pod import Pod

from utilities.constants import Containers, Ports

ProbeType = Literal["readinessProbe", "livenessProbe"]

VLLM_HEALTH_PATHS: tuple[str, ...] = ("/health", "/v1/models")

VLLM_READINESS_PROBE: dict[str, Any] = {
    "httpGet": {"path": "/health", "port": Ports.REST_PORT, "scheme": "HTTP"},
    "initialDelaySeconds": 120,
    "periodSeconds": 10,
    "timeoutSeconds": 10,
    "failureThreshold": 12,
}

VLLM_LIVENESS_PROBE: dict[str, Any] = {
    "httpGet": {"path": "/health", "port": Ports.REST_PORT, "scheme": "HTTP"},
    "initialDelaySeconds": 180,
    "periodSeconds": 30,
    "timeoutSeconds": 10,
    "failureThreshold": 10,
}


def get_kserve_container(pod: Pod) -> Any:
    """Return the kserve-container spec from the pod.

    Args:
        pod: Predictor pod for the vLLM InferenceService.

    Returns:
        Container spec object for kserve-container.

    Raises:
        ValueError: If kserve-container is not found in the pod spec.
    """
    for container in pod.instance.spec.containers:
        if container.name == Containers.KSERVE_CONTAINER_NAME:
            return container
    raise ValueError(f"{Containers.KSERVE_CONTAINER_NAME} not found in pod {pod.name}")


def get_optional_probe(pod: Pod, probe_type: ProbeType) -> dict[str, Any] | None:
    """Return probe configuration from kserve-container when present."""
    container = get_kserve_container(pod=pod)
    probe = getattr(container, probe_type, None)
    if not probe:
        return None
    return dict(probe)


def get_probe(pod: Pod, probe_type: ProbeType) -> dict[str, Any]:
    """Return the requested probe configuration from kserve-container.

    Args:
        pod: Predictor pod for the vLLM InferenceService.
        probe_type: Either readinessProbe or livenessProbe.

    Returns:
        Probe configuration dictionary.

    Raises:
        ValueError: If the requested probe is not configured on the container.
    """
    probe = get_optional_probe(pod=pod, probe_type=probe_type)
    if not probe:
        raise ValueError(f"{probe_type} not configured on {Containers.KSERVE_CONTAINER_NAME} in pod {pod.name}")
    return probe


def resolve_http_get(probe: dict[str, Any] | None, *, default_port: int = Ports.REST_PORT) -> dict[str, Any]:
    """Map a Kubernetes probe spec to an HTTP GET target for in-pod health checks.

    vLLM CUDA ServingRuntime templates may omit probes or use tcpSocket-only probes.
    When httpGet is absent, fall back to the standard vLLM REST port and /health path.
    """
    if probe:
        if http_get := probe.get("httpGet"):
            return dict(http_get)
        if tcp_socket := probe.get("tcpSocket"):
            port = tcp_socket.get("port", default_port)
            return {"path": "/health", "port": port, "scheme": "HTTP"}
    return {"path": "/health", "port": default_port, "scheme": "HTTP"}


def exec_http_probe(pod: Pod, http_get: dict[str, Any]) -> str:
    """Execute an HTTP GET probe inside the pod and return the status code.

    Args:
        pod: Predictor pod for the vLLM InferenceService.
        http_get: httpGet block from a readiness or liveness probe.

    Returns:
        HTTP status code as a string (e.g. "200").

    Raises:
        ValueError: If http_get is missing required fields.
    """
    path = http_get.get("path")
    port = http_get.get("port")
    if not path or port is None:
        raise ValueError(f"httpGet probe missing path or port: {http_get!r}")

    scheme = http_get.get("scheme", "HTTP")
    url = f"{'https' if scheme == 'HTTPS' else 'http'}://localhost:{port}{path}"
    curl_cmd = [
        "curl",
        "-s",
        "-o",
        "/dev/null",
        "-w",
        "%{http_code}",
        "--max-time",
        "15",
    ]
    if scheme == "HTTPS":
        curl_cmd.append("-k")
    curl_cmd.append(url)

    return pod.execute(command=curl_cmd, container=Containers.KSERVE_CONTAINER_NAME).strip()


def exec_vllm_health_check(pod: Pod, http_get: dict[str, Any]) -> str:
    """Try configured and fallback vLLM health paths until one returns HTTP 200."""
    paths = [http_get.get("path", "/health"), *VLLM_HEALTH_PATHS]
    unique_paths = list(dict.fromkeys(path for path in paths if path))
    last_status = "000"
    for path in unique_paths:
        last_status = exec_http_probe(pod=pod, http_get={**http_get, "path": path})
        if last_status == "200":
            return last_status
    return last_status
