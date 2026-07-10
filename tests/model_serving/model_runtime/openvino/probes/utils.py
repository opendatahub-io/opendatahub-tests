"""Utilities for OpenVINO (OVMS) readiness and liveness probe validation."""

from typing import Any, Literal

from ocp_resources.pod import Pod

from tests.model_serving.model_runtime.openvino.constant import OPENVINO_REST_PORT
from utilities.constants import Containers

ProbeType = Literal["readinessProbe", "livenessProbe"]

OVMS_HEALTH_PATHS: tuple[str, ...] = ("/v2/health/ready", "/v2/health/live")

OVMS_READINESS_PROBE: dict[str, Any] = {
    "httpGet": {"path": "/v2/health/ready", "port": OPENVINO_REST_PORT, "scheme": "HTTP"},
    "initialDelaySeconds": 30,
    "periodSeconds": 10,
    "timeoutSeconds": 5,
    "failureThreshold": 6,
}

OVMS_LIVENESS_PROBE: dict[str, Any] = {
    "httpGet": {"path": "/v2/health/live", "port": OPENVINO_REST_PORT, "scheme": "HTTP"},
    "initialDelaySeconds": 60,
    "periodSeconds": 30,
    "timeoutSeconds": 5,
    "failureThreshold": 5,
}


def get_kserve_container(pod: Pod) -> Any:
    for container in pod.instance.spec.containers:
        if container.name == Containers.KSERVE_CONTAINER_NAME:
            return container
    raise ValueError(f"{Containers.KSERVE_CONTAINER_NAME} not found in pod {pod.name}")


def get_probe(pod: Pod, probe_type: ProbeType) -> dict[str, Any]:
    container = get_kserve_container(pod=pod)
    probe = getattr(container, probe_type, None)
    if not probe:
        raise ValueError(f"{probe_type} not configured on {Containers.KSERVE_CONTAINER_NAME} in pod {pod.name}")
    return dict(probe)


def resolve_http_get(probe: dict[str, Any] | None, *, default_port: int = OPENVINO_REST_PORT) -> dict[str, Any]:
    if probe:
        if http_get := probe.get("httpGet"):
            return dict(http_get)
        if tcp_socket := probe.get("tcpSocket"):
            port = tcp_socket.get("port", default_port)
            return {"path": "/v2/health/ready", "port": port, "scheme": "HTTP"}
    return {"path": "/v2/health/ready", "port": default_port, "scheme": "HTTP"}


def exec_http_probe(pod: Pod, http_get: dict[str, Any]) -> str:
    path = http_get.get("path")
    port = http_get.get("port")
    if not path or port is None:
        raise ValueError(f"httpGet probe missing path or port: {http_get!r}")

    scheme = http_get.get("scheme", "HTTP")
    url = f"{'https' if scheme == 'HTTPS' else 'http'}://localhost:{port}{path}"
    curl_cmd = ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "15"]
    if scheme == "HTTPS":
        curl_cmd.append("-k")
    curl_cmd.append(url)

    return pod.execute(command=curl_cmd, container=Containers.KSERVE_CONTAINER_NAME).strip()


def exec_ovms_health_check(pod: Pod, http_get: dict[str, Any]) -> str:
    paths = [http_get.get("path", "/v2/health/ready"), *OVMS_HEALTH_PATHS]
    unique_paths = list(dict.fromkeys(path for path in paths if path))
    last_status = "000"
    for path in unique_paths:
        last_status = exec_http_probe(pod=pod, http_get={**http_get, "path": path})
        if last_status == "200":
            return last_status
    return last_status


def get_restart_counts(pod: Pod) -> dict[str, int]:
    return {container.name: container.restartCount for container in (pod.instance.status.containerStatuses or [])}


def pod_is_ready(pod: Pod) -> bool:
    for condition in pod.instance.status.conditions or []:
        if condition.type == "Ready":
            return condition.status == "True"
    return False
