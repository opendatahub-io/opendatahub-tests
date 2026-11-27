"""Helper functions for multi-model serving tests.

Consolidates pod-exec V2 protocol helpers and utility functions
used across E2E, negative, NFR, and upgrade test modules.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from ocp_resources.serving_runtime import ServingRuntime

from utilities.general import download_model_data
from utilities.infra import get_pods_by_isvc_label

LOGGER = structlog.get_logger(name=__name__)

# Container name constant shared across multi-model tests
KSERVE_CONTAINER: str = "kserve-container"

# Static-profile environment defaults recorded per RHAI-49122
STATIC_PROFILE_ENV_VARS: list[str] = [
    "CHECK_REPOSITORY_POPULATED",
    "MLSERVER_STRICT_READINESS",
    "MLSERVER_EMPTY_REGISTRY_READINESS",
    "MLSERVER_LOAD_MODELS_AT_STARTUP",
    "MLSERVER_MODELS_DIR",
    "MLSERVER_PARALLEL_WORKERS",
]


def append_env_to_serving_runtime(
    runtime: ServingRuntime,
    env_vars: list[dict[str, str]],
) -> None:
    """Append environment variables to the first container of a live ServingRuntime.

    Reads the current env list, appends any vars not already present, then
    sends a merge-patch with the full container array.  This avoids
    overwriting template-provided env vars (which the ``containers`` kwarg
    of ``ServingRuntimeFromTemplate`` would do).

    Args:
        runtime: An already-created ServingRuntime resource.
        env_vars: List of ``{"name": ..., "value": ...}`` dicts to append.
    """
    runtime.reload()
    containers = runtime.instance.spec.get("containers", [])
    if not containers:
        LOGGER.warning(
            "ServingRuntime has no containers; env patch skipped",
            runtime=runtime.name,
            namespace=runtime.namespace,
        )
        return

    existing_env = list(containers[0].get("env", []))
    existing_names = {e.get("name") for e in existing_env}
    for var in env_vars:
        if var["name"] not in existing_names:
            existing_env.append(var)

    containers[0]["env"] = existing_env
    runtime.update(resource_dict={"spec": {"containers": [dict(c) for c in containers]}})


def record_pod_image(pod: Pod) -> dict[str, str]:
    """Extract container image reference (repository, tag, digest) from a pod.

    Satisfies RHAI-49122 image traceability requirement.

    Args:
        pod: Running Kubernetes pod.

    Returns:
        Dictionary with repository, tag, digest, and full image string.
    """
    containers = pod.instance.spec.get("containers", [])
    if not containers:
        return {"image": "", "repository": "", "tag": "", "digest": ""}

    image: str = containers[0].get("image", "")
    repository = image
    tag = ""
    digest = ""

    if "@" in image:
        repository, digest = image.rsplit("@", 1)
    if ":" in repository:
        repository, tag = repository.rsplit(":", 1)

    # Also capture imageID from status for the digest that was actually pulled
    container_statuses = pod.instance.status.get("containerStatuses", [])
    if container_statuses:
        image_id = container_statuses[0].get("imageID", "")
        if "@" in image_id and not digest:
            digest = image_id.rsplit("@", 1)[1]

    return {"image": image, "repository": repository, "tag": tag, "digest": digest}


def record_static_profile_env(pod: Pod) -> dict[str, str | None]:
    """Record static-profile environment variable defaults from a deployed pod.

    Args:
        pod: Running Kubernetes pod.

    Returns:
        Dictionary mapping env var names to their values (None if unset).
    """
    env_values: dict[str, str | None] = {}
    containers = pod.instance.spec.get("containers", [])
    if not containers:
        return env_values

    env_list = containers[0].get("env", [])
    env_map = {e.get("name"): e.get("value") for e in env_list}
    for var_name in STATIC_PROFILE_ENV_VARS:
        env_values[var_name] = env_map.get(var_name)

    return env_values


def stage_s3_to_pvc(
    admin_client: DynamicClient,
    model_namespace: str,
    pvc_name: str,
    model_path: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    bucket_name: str,
    endpoint: str,
    region: str,
) -> str:
    """Download an S3 prefix into a PVC subdirectory.

    Thin wrapper around ``download_model_data`` so every staging fixture
    uses the same call convention.

    Args:
        admin_client: Kubernetes dynamic client.
        model_namespace: Namespace name.
        pvc_name: Target PVC name.
        model_path: S3 prefix **and** PVC subdirectory (they are the same
            by convention — ``download_model_data`` uses ``model_path``
            for both the source key-prefix and the ``subPath`` mount).
        aws_access_key_id: AWS access key.
        aws_secret_access_key: AWS secret key.
        bucket_name: S3 bucket name.
        endpoint: S3 endpoint URL.
        region: S3 region.

    Returns:
        ``model_path`` unchanged — callers build ``pvc://{pvc}/{path}``.
    """
    return download_model_data(
        client=admin_client,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        model_namespace=model_namespace,
        model_pvc_name=pvc_name,
        bucket_name=bucket_name,
        aws_endpoint_url=endpoint,
        aws_default_region=region,
        model_path=model_path,
        use_sub_path=True,
        restricted_scc_init=True,
    )


def get_isvc_pod(admin_client: DynamicClient, isvc: InferenceService) -> Pod:
    """Return the first pod for the given InferenceService."""
    pods = get_pods_by_isvc_label(client=admin_client, isvc=isvc)
    if not pods:
        raise RuntimeError(f"No pods found for InferenceService {isvc.name}")
    return pods[0]


def exec_in_pod(pod: Pod, command: list[str]) -> str:
    """Execute a command inside a pod and return stdout."""
    return pod.execute(command=command)


def v2_infer(pod: Pod, model_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Send a V2 inference request via pod exec curl.

    Raises RuntimeError when the HTTP status code cannot be parsed from the
    curl output, so callers never confuse a transport failure with a real
    server response.
    """
    url = f"http://localhost:8080/v2/models/{model_name}/infer"
    result = exec_in_pod(
        pod=pod,
        command=[
            "curl",
            "-s",
            "-w",
            "\n%{http_code}",
            "-X",
            "POST",
            url,
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(payload),
        ],
    )
    lines = result.strip().rsplit("\n", 1)
    if not lines[-1].isdigit():
        raise RuntimeError(f"Unparseable HTTP status from curl: {result!r}")
    body_str = lines[0] if len(lines) > 1 else ""
    http_code = int(lines[-1])
    try:
        body = json.loads(body_str) if body_str else {}
    except json.JSONDecodeError:
        body = {"raw": body_str}
    return {"http_code": http_code, "body": body}


def v2_health(pod: Pod, endpoint: str) -> int:
    """Query a V2 health endpoint and return the HTTP status code.

    Raises RuntimeError when the HTTP status code cannot be parsed, so
    callers never confuse a transport failure with a real probe result.
    """
    url = f"http://localhost:8080{endpoint}"
    result = exec_in_pod(
        pod=pod,
        command=["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", url],
    )
    status = result.strip()
    if not status.isdigit():
        raise RuntimeError(f"Unparseable HTTP status from health probe: {result!r}")
    return int(status)


def v2_repository_index(pod: Pod) -> list[dict[str, Any]]:
    """Query the V2 repository index and return the model list.

    Raises RuntimeError when the HTTP status is not 200 or the response
    is not a JSON list, so callers never confuse an error response with
    a valid (possibly empty) repository.
    """
    result = exec_in_pod(
        pod=pod,
        command=[
            "curl",
            "-s",
            "-w",
            "\n%{http_code}",
            "-X",
            "POST",
            "http://localhost:8080/v2/repository/index",
            "-H",
            "Content-Type: application/json",
            "-d",
            "{}",
        ],
    )
    lines = result.strip().rsplit("\n", 1)
    if not lines[-1].isdigit():
        raise RuntimeError(f"Unparseable HTTP status from repository index: {result!r}")
    http_code = int(lines[-1])
    body_str = lines[0] if len(lines) > 1 else ""
    if http_code != 200:
        raise RuntimeError(f"Repository index returned HTTP {http_code}: {body_str!r}")
    if not body_str:
        raise RuntimeError("Repository index returned HTTP 200 with empty body")
    try:
        parsed = json.loads(body_str)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Repository index returned invalid JSON: {body_str!r}") from exc
    if not isinstance(parsed, list):
        raise TypeError(f"Repository index returned {type(parsed).__name__}, expected list: {body_str!r}")
    for i, entry in enumerate(parsed):
        if not isinstance(entry, dict):
            raise TypeError(f"Repository index entry {i} is {type(entry).__name__}, expected dict: {entry!r}")
    return parsed


# ---------------------------------------------------------------------------
# Helpers consolidated from test modules
# ---------------------------------------------------------------------------


def get_pod_env_var(pod: Pod, var_name: str) -> str | None:
    """Extract an environment variable value from the pod spec.

    Raises RuntimeError if the pod has no containers, so callers never
    confuse an unreadable spec with 'env var not configured'.
    """
    containers = pod.instance.spec.get("containers", [])
    if not containers:
        raise RuntimeError(f"Pod {pod.name} has no containers in its spec")
    for container in containers:
        for env in container.get("env", []):
            if env.get("name") == var_name:
                return env.get("value")
    return None


def get_pod_probe(pod: Pod, probe_type: str) -> dict[str, Any] | None:
    """Extract a probe configuration from the pod spec.

    Raises RuntimeError if the pod has no containers, so callers never
    confuse an unreadable spec with 'probe not configured'.
    """
    containers = pod.instance.spec.get("containers", [])
    if not containers:
        raise RuntimeError(f"Pod {pod.name} has no containers in its spec")
    return containers[0].get(probe_type)


def v2_infer_timed(pod: Pod, model_name: str, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """Send a V2 inference request and return (result, server_latency_seconds).

    Uses curl's %{time_total} to measure server-side latency, excluding
    the Kubernetes exec session overhead.  ``--max-time 30`` bounds the
    in-pod curl so it completes before the Pod.execute timeout (60s),
    preventing ExecOnPodError from aborting the benchmark.  Curl timeouts
    emit HTTP code ``000`` which is returned as ``http_code=0``.
    """
    url = f"http://localhost:8080/v2/models/{model_name}/infer"
    result = exec_in_pod(
        pod=pod,
        command=[
            "curl",
            "-s",
            "--max-time",
            "30",
            "-w",
            "\n%{time_total}\n%{http_code}",
            "-X",
            "POST",
            url,
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(payload),
        ],
    )
    parts = result.strip().rsplit("\n", 2)
    if len(parts) == 3:
        body_str, time_total_str, http_code_str = parts
    elif len(parts) == 2:
        # Curl timeout: empty body stripped away, leaving only time + code
        body_str = ""
        time_total_str, http_code_str = parts
    else:
        raise RuntimeError(f"Unparseable curl output for timed infer: {result!r}")
    if not http_code_str.isdigit():
        raise RuntimeError(f"Unparseable HTTP status from timed infer: {result!r}")
    http_code = int(http_code_str)
    try:
        latency = float(time_total_str)
    except ValueError:
        raise RuntimeError(f"Unparseable time_total from timed infer: {time_total_str!r}")
    try:
        body = json.loads(body_str) if body_str else {}
    except json.JSONDecodeError:
        body = {"raw": body_str}
    return {"http_code": http_code, "body": body}, latency


def exec_curl_external(
    url: str,
    method: str = "POST",
    data: str | None = None,
) -> dict[str, Any]:
    """Execute a curl command from the test runner (external to cluster).

    Args:
        url: Target URL.
        method: HTTP method.
        data: Optional request body.

    Returns:
        Dictionary with http_code, returncode, and stderr.
    """
    curl_cmd: list[str] = [
        "curl",
        "-s",
        "-S",
        "-o",
        "/dev/null",
        "-w",
        "%{http_code}",
        "-X",
        method,
        "-H",
        "Content-Type: application/json",
        "--insecure",
        "--max-time",
        "10",
    ]
    if data:
        curl_cmd.extend(["-d", data])
    curl_cmd.append(url)

    try:
        result = subprocess.run(
            curl_cmd,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        code_str = result.stdout.strip()
        http_code = int(code_str) if code_str.isdigit() else 0
        return {
            "http_code": http_code,
            "returncode": result.returncode,
            "stderr": result.stderr.strip(),
        }
    except (subprocess.TimeoutExpired, ValueError) as exc:
        LOGGER.error("External curl failed", error=str(exc))
        return {"http_code": 0, "returncode": -1, "stderr": str(exc)}


def parse_cpu_millicores(value: str) -> int:
    """Convert a Kubernetes CPU string to integer millicores."""
    if value.endswith("m"):
        return int(value[:-1])
    return int(float(value) * 1000)


def parse_memory_mib(value: str) -> int:
    """Convert a Kubernetes memory string to integer MiB."""
    if value.endswith("Gi"):
        return int(float(value[:-2]) * 1024)
    if value.endswith("Mi"):
        return int(float(value[:-2]))
    if value.endswith("Ki"):
        return int(float(value[:-2]) / 1024)
    return int(int(value) / (1024 * 1024))


def collect_fleet_resource_requests(
    admin_client: DynamicClient,
    fleet: list[InferenceService],
) -> dict[str, Any]:
    """Collect aggregate resource requests for a fleet of ISVCs.

    Uses ``get_pods_by_isvc_label`` per ISVC instead of a label selector,
    because KServe raw-mode overwrites the ``app`` label on predictor pods.
    """
    metrics: dict[str, Any] = {
        "pod_count": 0,
        "total_cpu_millicores": 0,
        "total_memory_mib": 0,
    }
    for isvc in fleet:
        pods = get_pods_by_isvc_label(client=admin_client, isvc=isvc)
        metrics["pod_count"] += len(pods)
        for pod in pods:
            containers = pod.instance.spec.get("containers", [])
            for container in containers:
                requests = container.get("resources", {}).get("requests", {})
                cpu_str = requests.get("cpu", "0")
                mem_str = requests.get("memory", "0")
                metrics["total_cpu_millicores"] += parse_cpu_millicores(cpu_str)
                metrics["total_memory_mib"] += parse_memory_mib(mem_str)
    return metrics


def collect_pod_resource_requests(
    admin_client: DynamicClient,
    namespace: str,
    label_selector: str,
) -> dict[str, Any]:
    """Collect CPU and memory resource requests for pods matching a label.

    Args:
        admin_client: Kubernetes dynamic client.
        namespace: Namespace to query.
        label_selector: Label selector string.

    Returns:
        Dictionary with per-pod resources and aggregate totals.
    """
    pods = list(
        Pod.get(
            client=admin_client,
            namespace=namespace,
            label_selector=label_selector,
        )
    )
    metrics: dict[str, Any] = {
        "pod_count": len(pods),
        "total_cpu_millicores": 0,
        "total_memory_mib": 0,
    }
    for pod in pods:
        containers = pod.instance.spec.get("containers", [])
        for container in containers:
            requests = container.get("resources", {}).get("requests", {})
            cpu_str = requests.get("cpu", "0")
            mem_str = requests.get("memory", "0")
            metrics["total_cpu_millicores"] += parse_cpu_millicores(cpu_str)
            metrics["total_memory_mib"] += parse_memory_mib(mem_str)
    return metrics
