"""
Utility functions for LLM Deployment (LLMD) tests.

This module provides helper functions for LLMD test operations using ocp_resources.
Follows the established model server utils pattern for consistency.
"""

import json
from pathlib import Path

from kubernetes.dynamic import DynamicClient
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.pod import Pod
from ocp_resources.prometheus import Prometheus
from pyhelper_utils.shell import run_command
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutExpiredError, retry

from utilities.certificates_utils import get_ca_bundle
from utilities.constants import Timeout
from utilities.monitoring import get_metrics_value

LOGGER = get_logger(name=__name__)


def ns_from_file(file: str) -> str:
    """Derive namespace name from test filename.

    Example: __file__ of test_llmd_smoke.py → "llmd-smoke"
    """
    return Path(file).stem.removeprefix("test_").replace("_", "-")


def _collect_llmisvc_diagnostics(llmisvc: LLMInferenceService) -> str:
    """Collect diagnostic info from LLMISVC and its pods for timeout errors."""
    lines = [f"LLMInferenceService {llmisvc.name} in {llmisvc.namespace} did not become Ready."]

    # LLMISVC conditions
    conditions = llmisvc.instance.status.get("conditions", [])
    if conditions:
        lines.append("  Conditions:")
        lines.extend(f"    {condition['type']}: {condition['status']} — {condition.get('message', '')}" for condition in conditions)
    else:
        lines.append("  No conditions reported.")

    # Pod diagnostics
    client = llmisvc.client
    pods = get_llmd_workload_pods(client=client, llmisvc=llmisvc)
    if not pods:
        lines.append("  No workload pods found.")
    else:
        lines.append(f"  Workload pods ({len(pods)}):")
        for pod in pods:
            phase = pod.instance.status.phase
            lines.append(f"    {pod.name}: phase={phase}")
            for cs in pod.instance.status.get("containerStatuses", []):
                state = cs.get("state", {})
                if "waiting" in state:
                    lines.append(f"      {cs['name']}: Waiting — {state['waiting'].get('reason', '?')}")
                elif "terminated" in state:
                    lines.append(f"      {cs['name']}: Terminated — {state['terminated'].get('reason', '?')}")
                elif cs.get("restartCount", 0) > 0:
                    lines.append(f"      {cs['name']}: restarts={cs['restartCount']}")

    # Use oc CLI for events — Event.get() Python API can block on slow clusters
    try:
        _, stdout, _ = run_command(
            command=[
                "oc", "get", "events", "-n", llmisvc.namespace,
                "--field-selector", f"involvedObject.name={llmisvc.name}",
                "--sort-by=.lastTimestamp",
                "-o", "custom-columns=TYPE:.type,REASON:.reason,MESSAGE:.message",
                "--no-headers",
            ],
            verify_stderr=False, check=False,
        )
        if stdout.strip():
            lines.append("  Recent events:")
            for event_line in stdout.strip().splitlines()[-5:]:
                lines.append(f"    {event_line}")
    except Exception:  # noqa: BLE001
        LOGGER.debug("Failed to collect events for %s", llmisvc.name)

    return "\n".join(lines)


def wait_for_llmisvc(llmisvc: LLMInferenceService, timeout: int = Timeout.TIMEOUT_5MIN) -> None:
    """Wait for LLMISVC to reach Ready condition. Raises with diagnostics on timeout."""
    try:
        llmisvc.wait_for_condition(
            condition="Ready",
            status="True",
            timeout=timeout,
        )
        border = "=" * 60
        LOGGER.info(
            f"\n{border}\n  LLMInferenceService READY:"
            f"\n  Name: {llmisvc.name}"
            f"\n  Namespace: {llmisvc.namespace}"
            f"\n{border}"
        )
    except TimeoutExpiredError:
        diagnostics = _collect_llmisvc_diagnostics(llmisvc)
        LOGGER.error(diagnostics)
        raise TimeoutError(diagnostics)


def _get_inference_url(llmisvc: LLMInferenceService) -> str:
    """Extract inference URL from LLMISVC status."""
    status = llmisvc.instance.status
    if status and status.get("addresses"):
        addresses = status["addresses"]
        if addresses and addresses[0].get("url"):
            return addresses[0]["url"]
    if status and status.get("url"):
        return status["url"]
    return f"http://{llmisvc.name}.{llmisvc.namespace}.svc.cluster.local"


def _build_chat_body(model_name: str, prompt: str, max_tokens: int = 50) -> str:
    """Build OpenAI chat completion request body."""
    return json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    })


def _resolve_ca_cert(client: DynamicClient) -> str:
    """Get CA cert path for TLS verification. Returns path or empty string."""
    try:
        return get_ca_bundle(client=client, deployment_mode="raw")
    except Exception:  # noqa: BLE001
        return ""


def _log_curl_command(url: str, body: str, token: bool, ca_cert: str | None) -> None:
    """Log a human-readable curl command with token redacted and payload formatted."""
    formatted_body = json.dumps(json.loads(body), indent=2)
    auth_header = "\n  -H 'Authorization: Bearer ***REDACTED***'" if token else ""
    tls_flag = f"\n  --cacert {ca_cert}" if ca_cert else "\n  --insecure"
    LOGGER.info(
        f"curl -s -X POST \\\n"
        f"  -H 'Content-Type: application/json' \\\n"
        f"  -H 'Accept: application/json' \\{auth_header}\n"
        f"  -d '{formatted_body}' \\{tls_flag}\n"
        f"  {url}"
    )


def _curl_post(
    url: str, body: str, token: str | None = None, ca_cert: str | None = None, timeout: int = 60
) -> tuple[int, str]:
    """POST to URL via curl. Returns (status_code, response_body)."""
    cmd = [
        "curl", "-s", "-w", "\n%{http_code}",
        "-X", "POST",
        "-H", "Content-Type: application/json",
        "-H", "Accept: application/json",
        "-d", body,
        "--max-time", str(timeout),
    ]
    if token:
        cmd.extend(["-H", f"Authorization: Bearer {token}"])
    if ca_cert:
        cmd.extend(["--cacert", ca_cert])
    else:
        cmd.append("--insecure")
    cmd.append(url)

    _log_curl_command(url=url, body=body, token=bool(token), ca_cert=ca_cert)

    _, stdout, stderr = run_command(
        command=cmd, verify_stderr=False, check=False, hide_log_command=True
    )
    if not stdout.strip():
        raise ConnectionError(f"curl failed with no output: {stderr}")

    parts = stdout.rsplit("\n", 1)
    response_body = parts[0] if len(parts) > 1 else ""
    try:
        status_code = int(parts[-1].strip())
    except ValueError:
        status_code = 0
    return status_code, response_body


def _get_model_name(llmisvc: LLMInferenceService) -> str:
    """Read model name from spec.model.name, falling back to the resource name."""
    return llmisvc.instance.spec.model.get("name", llmisvc.name)


def send_chat_completions(
    llmisvc: LLMInferenceService,
    prompt: str,
    token: str | None = None,
    insecure: bool = True,
) -> tuple[int, str]:
    """Send a chat completion request. Returns (status_code, response_body)."""
    url = _get_inference_url(llmisvc) + "/v1/chat/completions"
    model_name = _get_model_name(llmisvc)
    body = _build_chat_body(model_name, prompt)
    ca_cert = None if insecure else _resolve_ca_cert(llmisvc.client)

    border = "=" * 60
    LOGGER.info(
        f"\n{border}\n  Sending inference request: {llmisvc.name}"
        f"\n  URL: {url}"
        f"\n  Model: {model_name}"
        f"\n{border}"
    )
    status_code, response_body = _curl_post(url, body, token=token, ca_cert=ca_cert)
    LOGGER.info(f"Inference response — status={status_code}\n{response_body}")
    return status_code, response_body


def parse_completion_text(response_body: str) -> str:
    """Extract completion text from a chat completion response."""
    data = json.loads(response_body)
    return data["choices"][0]["message"]["content"]


_CONTAINER_WAIT_ERRORS = {"InvalidImageName", "CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull"}
_CONTAINER_TERMINATED_ERRORS = {"Error", "CrashLoopBackOff"}


def _check_container_restarts(pod: Pod) -> dict[str, int]:
    """Return {container_name: restart_count} for containers with restarts > 0."""
    restarts: dict[str, int] = {}
    pod_status = pod.instance.status
    for cs in pod_status.get("containerStatuses", []) + pod_status.get("initContainerStatuses", []):
        if hasattr(cs, "restartCount") and cs.restartCount > 0:
            restarts[cs.name] = cs.restartCount
    return restarts


def _check_container_errors(pod: Pod) -> dict[str, str]:
    """Return {container_name: error_reason} for containers in error states."""
    errors: dict[str, str] = {}
    pod_status = pod.instance.status
    if not pod_status.containerStatuses:
        return errors
    for cs in pod_status.get("containerStatuses", []) + pod_status.get("initContainerStatuses", []):
        wait_state = cs.state.waiting
        term_state = cs.state.terminated
        if wait_state and wait_state.reason in _CONTAINER_WAIT_ERRORS:
            errors[cs.name] = wait_state.reason
        elif term_state and term_state.reason in _CONTAINER_TERMINATED_ERRORS:
            errors[cs.name] = term_state.reason
    return errors


def _check_pod_failure(pod: Pod) -> str | None:
    """Return failure reason if pod is in a failed phase, else None."""
    phase = pod.instance.status.phase
    if phase in (pod.Status.CRASH_LOOPBACK_OFF, pod.Status.FAILED):
        return phase
    return None



def assert_no_restarts(pods: list[Pod]) -> None:
    """Assert no container in any pod has restarted."""
    for pod in pods:
        restarts = _check_container_restarts(pod)
        assert not restarts, f"Pod {pod.name} has container restarts: {restarts}"


def assert_no_errors(pods: list[Pod]) -> None:
    """Assert no container is in an error state."""
    for pod in pods:
        errors = _check_container_errors(pod)
        assert not errors, f"Pod {pod.name} has container errors: {errors}"


def assert_no_failures(pods: list[Pod]) -> None:
    """Assert no pod is in a failed phase."""
    for pod in pods:
        failure = _check_pod_failure(pod)
        assert not failure, f"Pod {pod.name} is in {failure} phase"


def assert_pods_healthy(
    client: DynamicClient,
    llmisvc: LLMInferenceService,
) -> list[Pod]:
    """Assert workload pods are healthy (no restarts, errors, or failures). Returns pods."""
    pods = get_llmd_workload_pods(client=client, llmisvc=llmisvc)
    assert pods, f"No workload pods found for {llmisvc.name}"
    assert_no_restarts(pods)
    assert_no_errors(pods)
    assert_no_failures(pods)

    border = "=" * 60
    pod_names = "\n".join(f"    - {pod.name}" for pod in pods)
    LOGGER.info(
        f"\n{border}\n  Pods health OK ({len(pods)}):"
        f"\n{pod_names}"
        f"\n  Status: no restarts, no errors, no failures"
        f"\n{border}"
    )
    return pods


def get_llmd_workload_pods(
    client: DynamicClient,
    llmisvc: LLMInferenceService,
) -> list[Pod]:
    """
    Get all workload pods for an LLMInferenceService.

    Args:
        client: DynamicClient instance
        llmisvc: The LLMInferenceService to get pods for

    Returns:
        List of workload Pod objects
    """
    pods = []
    for pod in Pod.get(
        client=client,
        namespace=llmisvc.namespace,
        label_selector=(
            f"{Pod.ApiGroup.APP_KUBERNETES_IO}/part-of=llminferenceservice,"
            f"{Pod.ApiGroup.APP_KUBERNETES_IO}/name={llmisvc.name}"
        ),
    ):
        labels = pod.instance.metadata.get("labels", {})
        if labels.get("kserve.io/component") == "workload":
            pods.append(pod)
    return pods


def get_llmd_router_scheduler_pod(
    client: DynamicClient,
    llmisvc: LLMInferenceService,
) -> Pod | None:
    """
    Get the router-scheduler pod for an LLMInferenceService.

    Args:
        client: DynamicClient instance
        llmisvc: The LLMInferenceService to get router-scheduler pod for

    Returns:
        Router-scheduler Pod object or None if not found
    """
    for pod in Pod.get(
        client=client,
        namespace=llmisvc.namespace,
        label_selector=(
            f"{Pod.ApiGroup.APP_KUBERNETES_IO}/part-of=llminferenceservice,"
            f"{Pod.ApiGroup.APP_KUBERNETES_IO}/name={llmisvc.name}"
        ),
    ):
        labels = pod.instance.metadata.get("labels", {})
        if labels.get(f"{Pod.ApiGroup.APP_KUBERNETES_IO}/component") == "llminferenceservice-router-scheduler":
            return pod
    return None



def query_metric_by_pod(
    prometheus: Prometheus,
    metric_name: str,
    llmisvc: LLMInferenceService,
    pods: list[Pod],
) -> dict[str, float]:
    """Query a Prometheus metric for each pod. Returns {pod_name: value}."""
    result: dict[str, float] = {}
    for pod in pods:
        query = f'sum({metric_name}{{namespace="{llmisvc.namespace}",pod="{pod.name}"}})'
        result[pod.name] = float(get_metrics_value(prometheus=prometheus, metrics_query=query) or 0)
    return result


@retry(wait_timeout=90, sleep=30, exceptions_dict={AssertionError: []}, print_log=False)
def assert_prefix_cache_routing(
    prometheus: Prometheus,
    llmisvc: LLMInferenceService,
    pods: list[Pod],
    expected_requests: int,
    block_size: int = 64,
) -> bool:
    """Assert all traffic routed to 1 pod with correct cache hits. Retries for metric delay."""
    requests = query_metric_by_pod(prometheus, "kserve_vllm:request_success_total", llmisvc, pods)
    LOGGER.info(f"Request count by pod: {requests}")

    pods_with_traffic = [p for p, count in requests.items() if count > 0]
    assert len(pods_with_traffic) == 1, (
        f"Expected traffic on exactly 1 pod, got {len(pods_with_traffic)}: {requests}"
    )

    active_pod = pods_with_traffic[0]
    assert requests[active_pod] == expected_requests, (
        f"Expected {expected_requests} requests on '{active_pod}', got {requests[active_pod]}"
    )

    hits = query_metric_by_pod(prometheus, "kserve_vllm:prefix_cache_hits_total", llmisvc, pods)
    LOGGER.info(f"Prefix cache hits by pod: {hits}")

    expected_hits = (expected_requests - 1) * block_size
    assert hits[active_pod] == expected_hits, (
        f"Expected {expected_hits} cache hits on '{active_pod}', got {hits[active_pod]}"
    )
    return True


@retry(wait_timeout=90, sleep=30, exceptions_dict={AssertionError: []}, print_log=False)
def assert_scheduler_routing(router_pod: Pod, min_decisions: int) -> bool:
    """Assert scheduler made enough routing decisions. Retries for log propagation."""
    logs = get_scheduler_decision_logs(router_scheduler_pod=router_pod)
    assert len(logs) >= min_decisions, (
        f"Expected >= {min_decisions} scheduler decisions, got {len(logs)}"
    )
    return True


def send_prefix_cache_requests(
    llmisvc: LLMInferenceService,
    prompt: str,
    token: str,
    count: int = 20,
    min_ratio: float = 0.8,
) -> int:
    """Send identical requests for prefix cache testing. Returns success count."""
    LOGGER.info(f"Sending {count} identical requests to test prefix cache")
    successful = 0
    for i in range(count):
        try:
            status, _ = send_chat_completions(
                llmisvc, prompt=prompt, token=token, insecure=False
            )
            if status == 200:
                successful += 1
        except Exception as e:  # noqa: BLE001
            LOGGER.error(f"Request {i + 1}/{count} failed: {e}")
    LOGGER.info(f"{successful}/{count} requests succeeded")
    assert successful >= count * min_ratio, (
        f"Too many failures: {successful}/{count} (need {min_ratio * 100}%)"
    )
    return successful


def get_scheduler_decision_logs(
    router_scheduler_pod: Pod,
    lookback_seconds: int = 600,
) -> list[dict]:
    """
    Retrieve scheduling decision logs from the router-scheduler pod.

    Args:
        router_scheduler_pod: The router-scheduler Pod object
        lookback_seconds: How far back to look in logs (default: 600s = 10 minutes)

    Returns:
        list[dict]: List of parsed JSON log entries containing scheduler decisions
    """
    LOGGER.info(f"Retrieving logs from scheduler pod {router_scheduler_pod.name}")

    # Get all logs from the scheduler pod
    # Note: The router-scheduler container is the default/main container
    raw_logs = router_scheduler_pod.log()

    # Target decision message
    target_decision_msg = "Selecting pods from candidates sorted by max score"

    # Filtering logs
    filtered_logs = "\n".join(line for line in raw_logs.splitlines() if target_decision_msg in line)

    # Parsing as json
    json_logs = [json.loads(line) for line in filtered_logs.splitlines()]

    LOGGER.info(f"Retrieved {len(json_logs)} logs from router-scheduler pod")
    return json_logs


