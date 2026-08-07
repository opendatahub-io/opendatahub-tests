"""Utilities for Triton readiness and liveness probe validation."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Literal

from kubernetes.dynamic import DynamicClient
from ocp_resources.exceptions import ExecOnPodError
from ocp_resources.pod import Pod
from ocp_resources.template import Template
from pytest_testconfig import config as py_config

from tests.model_serving.model_runtime.triton.constant import TRITON_REST_PORT
from utilities.constants import Containers, Protocols

ProbeType = Literal["readinessProbe", "livenessProbe"]

TRITON_HEALTH_PATHS: tuple[str, ...] = ("/v2/health/live", "/v2/health/ready")

TRITON_READINESS_PROBE: dict[str, Any] = {
    "httpGet": {"path": "/v2/health/ready", "port": TRITON_REST_PORT, "scheme": "HTTP"},
    "initialDelaySeconds": 120,  # Increased for larger models
    "periodSeconds": 10,
    "timeoutSeconds": 10,
    "failureThreshold": 30,  # Increased: 120s initial + (30 failures * 10s period) = up to 420s total
}

TRITON_LIVENESS_PROBE: dict[str, Any] = {
    "httpGet": {"path": "/v2/health/live", "port": TRITON_REST_PORT, "scheme": "HTTP"},
    "initialDelaySeconds": 180,  # Increased for larger models
    "periodSeconds": 30,
    "timeoutSeconds": 10,
    "failureThreshold": 20,  # Increased: allows up to 600s after initial delay
}


def get_kserve_container(pod: Pod) -> Any:
    """Return the kserve-container spec from the pod.

    Args:
        pod: Predictor pod for the Triton InferenceService.

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
        pod: Predictor pod for the Triton InferenceService.
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


def resolve_http_get(probe: dict[str, Any] | None, *, default_port: int = TRITON_REST_PORT) -> dict[str, Any]:
    """Map a Kubernetes probe spec to an HTTP GET target for in-pod health checks.

    Triton ServingRuntime templates may omit probes or use tcpSocket-only probes.
    When httpGet is absent, fall back to the standard Triton REST port and /v2/health/live path.
    """
    if probe:
        if http_get := probe.get("httpGet"):
            return dict(http_get)
        if tcp_socket := probe.get("tcpSocket"):
            port = tcp_socket.get("port", default_port)
            return {"path": "/v2/health/live", "port": port, "scheme": "HTTP"}
    return {"path": "/v2/health/live", "port": default_port, "scheme": "HTTP"}


def exec_http_probe(pod: Pod, http_get: dict[str, Any]) -> str:
    """Execute an HTTP GET probe inside the pod and return the status code.

    Args:
        pod: Predictor pod for the Triton InferenceService.
        http_get: httpGet block from a readiness or liveness probe.

    Returns:
        HTTP status code as a string (e.g. "200").

    Raises:
        ValueError: If http_get is missing required fields.
        RuntimeError: If pod execution fails.
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

    try:
        result = pod.execute(command=curl_cmd, container=Containers.KSERVE_CONTAINER_NAME)
        return result.strip()
    except ExecOnPodError as e:
        raise RuntimeError(f"Failed to execute probe in pod {pod.name}: {e}") from e


def exec_triton_health_check(pod: Pod, http_get: dict[str, Any]) -> str:
    """Try configured and fallback Triton health paths until one returns HTTP 200."""
    paths = [http_get.get("path", "/v2/health/live"), *TRITON_HEALTH_PATHS]
    unique_paths = list(dict.fromkeys(path for path in paths if path))
    last_status = "000"
    for path in unique_paths:
        last_status = exec_http_probe(pod=pod, http_get={**http_get, "path": path})
        if last_status == "200":
            return last_status
    return last_status


@contextmanager
def create_triton_template(
    admin_client: DynamicClient, protocol: str, triton_runtime_image: str
) -> Generator[Template, Any, Any]:
    """Create a Triton ServingRuntime template."""
    template_dict = {
        "apiVersion": "template.openshift.io/v1",
        "kind": "Template",
        "metadata": {
            "name": f"triton-{protocol}-runtime-template",
            "namespace": py_config["applications_namespace"],
        },
        "objects": [create_triton_serving_runtime(protocol=protocol, triton_runtime_image=triton_runtime_image)],
        "parameters": [],
    }

    with Template(
        client=admin_client,
        namespace=py_config["applications_namespace"],
        kind_dict=template_dict,
    ) as template:
        yield template


def create_triton_serving_runtime(protocol: str, triton_runtime_image: str) -> dict[str, Any]:
    """Create Triton ServingRuntime object definition."""
    volumes = []
    volume_mounts = []
    if protocol == Protocols.GRPC:
        volumes.append({"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"}})
        volume_mounts.append({"name": "shm", "mountPath": "/dev/shm"})

    port_config = {
        "name": "h2c" if protocol == Protocols.GRPC else "http1",
        "containerPort": 9000 if protocol == Protocols.GRPC else 8080,
        "protocol": "TCP",
    }

    container_args = [
        "tritonserver",
        "--model-store=/mnt/models",
        f"--{'grpc' if protocol == Protocols.GRPC else 'http'}-port={port_config['containerPort']}",
        f"--{'allow-grpc' if protocol == Protocols.GRPC else 'allow-http'}=True",
    ]

    kserve_container: list[dict[str, Any]] = [
        {
            "name": "kserve-container",
            "image": triton_runtime_image,
            "args": container_args,
            "ports": [port_config],
            "volumeMounts": volume_mounts,
            "resources": {
                "requests": {"cpu": "1", "memory": "2Gi"},
                "limits": {"cpu": "1", "memory": "2Gi"},
            },
        }
    ]

    return {
        "apiVersion": "serving.kserve.io/v1alpha1",
        "kind": "ServingRuntime",
        "metadata": {
            "name": f"triton-{protocol}-runtime",
            "annotations": {
                "prometheus.kserve.io/path": "/metrics",
                "prometheus.kserve.io/port": "8002",
            },
        },
        "spec": {
            "containers": kserve_container,
            "volumes": volumes,
            "protocolVersions": ["v2", "grpc-v2"],
            "supportedModelFormats": [
                {"name": "tensorrt", "version": "8", "autoSelect": True, "priority": 1},
                {"name": "tensorflow", "version": "1", "autoSelect": True, "priority": 1},
                {"name": "tensorflow", "version": "2", "autoSelect": True, "priority": 1},
                {"name": "onnx", "version": "1", "autoSelect": True, "priority": 1},
                {"name": "pytorch", "version": "1", "autoSelect": True},
                {"name": "triton", "version": "2", "autoSelect": True, "priority": 1},
                {"name": "xgboost", "version": "1", "autoSelect": True},
                {"name": "python", "version": "1", "autoSelect": True},
            ],
        },
    }
