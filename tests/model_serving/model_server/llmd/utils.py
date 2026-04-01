"""
Utility functions for LLM Deployment (LLMD) tests.

This module provides helper functions for LLMD test operations using ocp_resources.
Follows the established model server utils pattern for consistency.
"""

import json
from pathlib import Path

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.pod import Pod
from ocp_resources.prometheus import Prometheus
from ocp_resources.route import Route
from pyhelper_utils.shell import run_command
from timeout_sampler import retry

from utilities.certificates_utils import get_ca_bundle
from utilities.constants import Timeout
from utilities.infra import is_disconnected_cluster
from utilities.llmd_constants import LLMDGateway, LLMEndpoint
from utilities.monitoring import get_metrics_value

LOGGER = structlog.get_logger(name=__name__)


def ns_from_file(file: str) -> str:
    """Derive namespace name from test filename.

    Example: __file__ of test_llmd_smoke.py → "llmd-smoke"
    """
    return Path(file).stem.removeprefix("test_").replace("_", "-")[:63]


def wait_for_llmisvc(llmisvc: LLMInferenceService, timeout: int = Timeout.TIMEOUT_5MIN) -> None:
    """Wait for LLMISVC to reach Ready condition. Raises on timeout."""
    llmisvc.wait_for_condition(
        condition="Ready",
        status="True",
        timeout=timeout,
    )
    LOGGER.info(f"LLMInferenceService {llmisvc.name} is Ready in namespace {llmisvc.namespace}")


def wait_for_llmisvc_pods_ready(
    client: DynamicClient,
    llmisvc: LLMInferenceService,
    timeout: int = 30,
) -> None:
    """Wait for all LLMISVC pods (workload + router-scheduler) to be Ready."""
    pods = list(
        Pod.get(
            client=client,
            namespace=llmisvc.namespace,
            label_selector=(
                f"{Pod.ApiGroup.APP_KUBERNETES_IO}/part-of=llminferenceservice,"
                f"{Pod.ApiGroup.APP_KUBERNETES_IO}/name={llmisvc.name}"
            ),
        )
    )
    LOGGER.info(f"Waiting for {len(pods)} pod(s) to be Ready for {llmisvc.name}")
    for pod in pods:
        pod.wait_for_condition(condition="Ready", status="True", timeout=timeout)
        LOGGER.info(f"Pod {pod.name} is Ready")


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


def _get_disconnected_inference_url(llmisvc: LLMInferenceService) -> str:
    """Build inference URL using the gateway Route for disconnected clusters.

    On disconnected clusters the gateway uses ClusterIP instead of LoadBalancer,
    so the internal service URL from LLMISVC status is not reachable from outside
    the cluster. This function resolves the URL via the gateway Route instead.
    """
    route = Route(
        client=llmisvc.client,
        name=LLMDGateway.DEFAULT_NAME,
        namespace=LLMDGateway.DEFAULT_NAMESPACE,
    )
    if not route.exists:
        raise RuntimeError(
            f"Gateway Route {LLMDGateway.DEFAULT_NAME} not found in {LLMDGateway.DEFAULT_NAMESPACE}. "
            "Disconnected clusters require the gateway Route to be configured."
        )
    host = route.instance.spec.host
    if not host:
        raise RuntimeError(
            f"Gateway Route {LLMDGateway.DEFAULT_NAME} in {LLMDGateway.DEFAULT_NAMESPACE} "
            "has no host set. Ensure the Route is fully configured."
        )
    return f"https://{host}/{llmisvc.namespace}/{llmisvc.name}"


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
        "curl",
        "-s",
        "-w",
        "\n%{http_code}",
        "-X",
        "POST",
        "-H",
        "Content-Type: application/json",
        "-H",
        "Accept: application/json",
        "-d",
        body,
        "--max-time",
        str(timeout),
    ]
    if token:
        cmd.extend(["-H", f"Authorization: Bearer {token}"])
    if ca_cert:
        cmd.extend(["--cacert", ca_cert])
    else:
        cmd.append("--insecure")
    cmd.append(url)

    _log_curl_command(url=url, body=body, token=bool(token), ca_cert=ca_cert)

    _, stdout, stderr = run_command(command=cmd, verify_stderr=False, check=False, hide_log_command=True)
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
    base_url = (
        _get_disconnected_inference_url(llmisvc)
        if is_disconnected_cluster(llmisvc.client)
        else _get_inference_url(llmisvc)
    )
    url = base_url + LLMEndpoint.CHAT_COMPLETIONS
    model_name = _get_model_name(llmisvc=llmisvc)
    body = _build_chat_body(model_name=model_name, prompt=prompt)
    ca_cert = None if insecure else _resolve_ca_cert(llmisvc.client)

    LOGGER.info(f"Sending inference request to {llmisvc.name} — URL: {url}, Model: {model_name}")
    status_code, response_body = _curl_post(url=url, body=body, token=token, ca_cert=ca_cert)
    LOGGER.info(f"Inference response — status={status_code}\n{response_body}")
    return status_code, response_body


def parse_completion_text(response_body: str) -> str:
    """Extract completion text from a chat completion response."""
    try:
        data = json.loads(response_body)
        return data["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Failed to parse completion response: {e}\nBody: {response_body[:500]}") from e


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
    requests = query_metric_by_pod(
        prometheus=prometheus,
        metric_name="kserve_vllm:request_success_total",
        llmisvc=llmisvc,
        pods=pods,
    )
    LOGGER.info(f"Request count by pod: {requests}")

    pods_with_traffic = [p for p, count in requests.items() if count > 0]
    assert len(pods_with_traffic) == 1, f"Expected traffic on exactly 1 pod, got {len(pods_with_traffic)}: {requests}"

    active_pod = pods_with_traffic[0]
    assert requests[active_pod] == expected_requests, (
        f"Expected {expected_requests} requests on '{active_pod}', got {requests[active_pod]}"
    )

    hits = query_metric_by_pod(
        prometheus=prometheus,
        metric_name="kserve_vllm:prefix_cache_hits_total",
        llmisvc=llmisvc,
        pods=pods,
    )
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
    assert len(logs) >= min_decisions, f"Expected >= {min_decisions} scheduler decisions, got {len(logs)}"
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
                llmisvc=llmisvc,
                prompt=prompt,
                token=token,
                insecure=False,
            )
            if status == 200:
                successful += 1
        except Exception:
            LOGGER.exception(f"Request {i + 1}/{count} failed")
    LOGGER.info(f"{successful}/{count} requests succeeded")
    assert successful >= count * min_ratio, f"Too many failures: {successful}/{count} (need {min_ratio * 100}%)"
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
    raw_logs = router_scheduler_pod.log(container="main")

    # Target decision message
    target_decision_msg = "Selecting endpoints from candidates sorted by max score"

    # Filtering logs
    filtered_logs = "\n".join(line for line in raw_logs.splitlines() if target_decision_msg in line)

    # Parsing as json
    json_logs = [json.loads(line) for line in filtered_logs.splitlines()]

    LOGGER.info(f"Retrieved {len(json_logs)} logs from router-scheduler pod")
    return json_logs
