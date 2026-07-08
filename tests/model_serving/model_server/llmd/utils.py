"""
Utility functions for LLM Deployment (LLMD) tests.

This module provides helper functions for LLMD test operations using ocp_resources.
Follows the established model server utils pattern for consistency.
"""

import json
import re
import time
from pathlib import Path
from typing import Any

import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.event import Event
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.node import Node
from ocp_resources.pod import Pod
from ocp_resources.prometheus import Prometheus
from pyhelper_utils.shell import run_command
from timeout_sampler import TimeoutExpiredError, retry

from tests.model_serving.model_server.llmd.constants import LLMD_TESTS_SUPPORTED_ACCELERATORS
from utilities.certificates_utils import get_ca_bundle
from utilities.jira import is_jira_issue_open
from utilities.llmd_constants import LLMEndpoint
from utilities.llmd_utils import get_llm_inference_url
from utilities.monitoring import get_metrics_value

LOGGER = structlog.get_logger(name=__name__)


def detect_accelerators(client: DynamicClient) -> list[dict[str, int]]:
    """Detect accelerator resources available on cluster worker nodes.

    Returns:
        List of dicts, one per accelerator node. Each dict maps accelerator
        resource name to available count.
        Example: [{"amd.com/gpu": 8}, {"amd.com/gpu": 4}]
    """
    accelerators: list[dict[str, int]] = []
    for node in Node.get(client=client, label_selector="node-role.kubernetes.io/worker"):
        allocatable = node.instance.status.allocatable or {}
        node_accelerators = {
            resource: int(allocatable[resource])
            for resource in LLMD_TESTS_SUPPORTED_ACCELERATORS
            if int(allocatable.get(resource, 0)) > 0
        }
        if node_accelerators:
            LOGGER.info(f"[llmd] Accelerator node {node.name}: {node_accelerators}")
            accelerators.append(node_accelerators)

    return accelerators


class AcceleratorConfigDiscoveryResult:
    """Result of an LLMInferenceServiceConfig CR discovery attempt.

    Attributes:
        name: The name of the matched CR, or None if no CR matched all criteria.
        all_cr_names: Every CR name found in the namespace (for diagnostics).
        candidates: Human-readable descriptions of CRs that passed the name regex
            filter, including their annotation values (for skip/error messages).
    """

    def __init__(self, name: str | None, all_cr_names: list[str], candidates: list[str]):
        self.name = name
        self.all_cr_names = all_cr_names
        self.candidates = candidates


def list_matching_accelerator_configs(
    client: DynamicClient,
    namespace: str,
    accelerator: str,
    topology: str,
    name_regex: str = "",
) -> AcceleratorConfigDiscoveryResult:
    """Find an LLMInferenceServiceConfig CR matching accelerator, topology, and optional name regex.

    Lists LLMInferenceServiceConfig CRs in the given namespace and filters by
    ``opendatahub.io/recommended-accelerators`` and
    ``opendatahub.io/supported-topologies`` annotations.  When ``name_regex``
    is non-empty, only CRs whose name matches the regex are considered.

    Args:
        client: Kubernetes dynamic client.
        namespace: The namespace where LLMInferenceServiceConfig CRs are deployed
            (typically the DSCI applications namespace).
        accelerator: The k8s accelerator resource name (e.g. ``nvidia.com/gpu``).
        topology: The deployment topology to match (e.g. ``workload-single-node``).
        name_regex: Optional regex to filter CR names (e.g. ``.*fast-1$``).

    Returns:
        AcceleratorConfigDiscoveryResult with the matched CR name (or None), all CR names in
        the namespace, and details on candidate CRs for diagnostics.
    """
    try:
        api = client.resources.get(
            api_version="serving.kserve.io/v1alpha1",
            kind="LLMInferenceServiceConfig",
        )
        all_crs = api.get(namespace=namespace)
    except ResourceNotFoundError:
        LOGGER.warning("[llmd] LLMInferenceServiceConfig CRD not found on cluster")
        return AcceleratorConfigDiscoveryResult(name=None, all_cr_names=[], candidates=[])

    all_cr_names = [cr.metadata.name for cr in all_crs.items]
    LOGGER.info(f"[llmd] LLMInferenceServiceConfig CRs in '{namespace}': {all_cr_names}")

    candidates = []
    for cr in all_crs.items:
        name = cr.metadata.name
        if name_regex and not re.search(name_regex, name):
            continue
        annotations = cr.metadata.get("annotations") or {}
        raw_accel = annotations.get("opendatahub.io/recommended-accelerators", "[]")
        raw_topo = annotations.get("opendatahub.io/supported-topologies", "[]")
        try:
            recommended = json.loads(raw_accel)
            topologies = json.loads(raw_topo)
        except (ValueError, TypeError):  # fmt: skip
            candidates.append(f"{name} (unparseable annotations)")
            continue
        candidates.append(f"{name} (accelerators={recommended}, topologies={topologies})")
        if accelerator in recommended and topology in topologies:
            LOGGER.info(
                f"[llmd] Matched CR: {name} (accelerator={accelerator}, topology={topology}, regex='{name_regex}')"
            )
            return AcceleratorConfigDiscoveryResult(name=name, all_cr_names=all_cr_names, candidates=candidates)

    LOGGER.warning(
        f"[llmd] No LLMInferenceServiceConfig CR matched all criteria."
        f" Searched namespace='{namespace}' for CRs with:"
        f" opendatahub.io/recommended-accelerators containing '{accelerator}',"
        f" opendatahub.io/supported-topologies containing '{topology}',"
        f" name matching regex '{name_regex}'."
        f" CRs that passed the name regex filter (with their annotations):"
        f" {candidates or 'none'}."
        f" All LLMInferenceServiceConfig CRs in '{namespace}': {all_cr_names or 'none'}"
    )
    return AcceleratorConfigDiscoveryResult(name=None, all_cr_names=all_cr_names, candidates=candidates)


def ns_from_file(file: str) -> str:
    """Derive namespace name from test filename.

    Example: __file__ of test_llmd_smoke.py → "llmd-smoke"
    """
    return Path(file).stem.removeprefix("test_").replace("_", "-")[:63]


def _debug_info_conditions(llmisvc: LLMInferenceService) -> str:
    """Return debug info containing LLMISVC status conditions."""
    conditions = llmisvc.instance.status.get("conditions", [])
    lines = []
    for condition in conditions:
        line = f"  * {condition['type']}: {condition['status']}"
        if condition.get("reason"):
            line += f" reason={condition['reason']}"
        if condition.get("message"):
            line += f" message={condition['message']}"
        lines.append(line)
    return "\n".join(lines) or "  (no conditions)"


def _debug_info_pod_statuses(llmisvc: LLMInferenceService) -> str:
    """Return debug info containing pod phase, restart count, and waiting reasons."""
    pods = list(
        Pod.get(
            client=llmisvc.client,
            namespace=llmisvc.namespace,
            label_selector=(
                f"{Pod.ApiGroup.APP_KUBERNETES_IO}/part-of=llminferenceservice,"
                f"{Pod.ApiGroup.APP_KUBERNETES_IO}/name={llmisvc.name}"
            ),
        )
    )
    if not pods:
        return "  (no pods found)"

    lines = []
    for pod in pods:
        phase = pod.instance.status.phase
        all_statuses = (pod.instance.status.get("initContainerStatuses") or []) + (
            pod.instance.status.get("containerStatuses") or []
        )
        restarts = sum(container_status.get("restartCount", 0) for container_status in all_statuses)
        parts = [f"* pod={pod.name} phase={phase} restarts={restarts}"]

        for container_status in all_statuses:
            state = container_status.get("state") or {}
            waiting = state.get("waiting")
            if waiting:
                # Container is currently waiting (e.g. CrashLoopBackOff, ImagePullBackOff)
                reason = waiting.get("reason", "Unknown")
                message = waiting.get("message", "")
                parts.append(f"{reason}" + (f": {message}" if message else ""))
            elif container_status.get("restartCount", 0) > 0:
                # Container is running but has restarted — show why it last crashed
                terminated = (container_status.get("lastState") or {}).get("terminated")
                if terminated:
                    parts.append(
                        f" {container_status['name']}: last terminated"
                        f" reason={terminated.get('reason', 'Unknown')}"
                        f" exitCode={terminated.get('exitCode', '?')}"
                    )

        lines.append("  " + " | ".join(parts))
    return "\n".join(lines)


def _debug_info_events(llmisvc: LLMInferenceService) -> str:
    """Collect recent warning events from the LLMISVC namespace."""
    events = Event.list(
        client=llmisvc.client,
        namespace=llmisvc.namespace,
        field_selector="type=Warning",
        since_seconds=600,
    )
    if not events:
        return "  (no warning events)"

    lines = []
    for event in events:
        timestamp = str(event.get("lastTimestamp") or event.get("eventTime") or "")
        if "T" in timestamp:
            timestamp = timestamp.split("T")[1][:8]
        reason = event.get("reason", "")
        obj = event.get("involvedObject") or {}
        obj_name = obj.get("name", "")
        msg = " ".join(event.get("message", "").split())
        count = event.get("count", 1)
        count_str = f" (x{count})" if count and count > 1 else ""
        lines.append(f"  * {reason}{count_str} — {msg} [{obj_name}][{timestamp}]")
    return "\n".join(lines)


def _log_llmisvc_debug_info(llmisvc: LLMInferenceService) -> None:
    """Log debug info related to LLMISVC timeout: conditions, pod statuses, and events."""
    name, ns = llmisvc.name, llmisvc.namespace
    separator = "=" * 60
    sections = [
        f"\n{separator}",
        f"  LLMISVC {name} timed out in {ns}",
        separator,
    ]
    for label, func in [
        ("Conditions", lambda: _debug_info_conditions(llmisvc)),
        ("Pods", lambda: _debug_info_pod_statuses(llmisvc)),
        ("Events", lambda: _debug_info_events(llmisvc)),
    ]:
        try:
            sections.append(f"\n {label}:\n{func()}")
        except Exception:  # noqa: BLE001
            sections.append(f"\n {label}:\n  (failed to collect)")
    sections.append(separator + "\n")
    LOGGER.error("\n".join(sections))


def wait_for_llmisvc(llmisvc: LLMInferenceService, timeout: int = 300) -> None:
    """Wait for LLMISVC to reach Ready condition. Raises on timeout."""
    try:
        llmisvc.wait_for_condition(
            condition="Ready",
            status="True",
            timeout=timeout,
        )
    except TimeoutExpiredError:
        _log_llmisvc_debug_info(llmisvc)
        raise
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


def _build_chat_body(model_name: str, prompt: str, max_tokens: int = LLMEndpoint.DEFAULT_MAX_TOKENS) -> str:
    """Build OpenAI chat completion request body."""
    return json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": LLMEndpoint.DEFAULT_TEMPERATURE,
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


def _curl_request(
    method: str,
    url: str,
    body: str | None = None,
    token: str | None = None,
    ca_cert: str | None = None,
    timeout: int = LLMEndpoint.DEFAULT_TIMEOUT,
) -> tuple[int, str]:
    """Execute an HTTP request via curl. Returns (status_code, response_body)."""
    cmd = [
        "curl",
        "-s",
        "-w",
        "\n%{http_code}",
        "-H",
        "Accept: application/json",
        "--max-time",
        str(timeout),
    ]
    if body is not None:
        cmd.extend(["-X", "POST", "-H", "Content-Type: application/json", "-d", body])
    if token:
        cmd.extend(["-H", f"Authorization: Bearer {token}"])
    if ca_cert:
        cmd.extend(["--cacert", ca_cert])
    else:
        cmd.append("--insecure")
    cmd.append(url)

    if body is not None:
        _log_curl_command(url=url, body=body, token=bool(token), ca_cert=ca_cert)
    else:
        LOGGER.info(f"{method} {url}")

    _, stdout, stderr = run_command(command=cmd, verify_stderr=False, check=False, hide_log_command=True)
    if not stdout.strip():
        raise ConnectionError(f"curl {method} failed with no output: {stderr}")

    parts = stdout.rsplit("\n", 1)
    response_body = parts[0] if len(parts) > 1 else ""
    try:
        status_code = int(parts[-1].strip())
    except ValueError:
        status_code = 0
    return status_code, response_body


def _curl_post(
    url: str,
    body: str,
    token: str | None = None,
    ca_cert: str | None = None,
    timeout: int = LLMEndpoint.DEFAULT_TIMEOUT,
) -> tuple[int, str]:
    """POST to URL via curl. Returns (status_code, response_body)."""
    return _curl_request(method="POST", url=url, body=body, token=token, ca_cert=ca_cert, timeout=timeout)


def _curl_get(
    url: str,
    token: str | None = None,
    ca_cert: str | None = None,
    timeout: int = LLMEndpoint.DEFAULT_TIMEOUT,
) -> tuple[int, str]:
    """GET a URL via curl. Returns (status_code, response_body)."""
    return _curl_request(method="GET", url=url, token=token, ca_cert=ca_cert, timeout=timeout)


def get_vllm_version(
    llmisvc: LLMInferenceService,
    token: str | None = None,
    insecure: bool = True,
) -> str:
    """Query the vLLM /version endpoint and return the version string.

    Args:
        llmisvc: The LLMInferenceService to query.
        token: Optional bearer token for authentication.
        insecure: Skip TLS verification (default True).

    Returns:
        The vLLM version string (e.g. "0.8.5").

    Raises:
        ValueError: If the response cannot be parsed or the endpoint returns an error.
    """
    base_url = get_llm_inference_url(llm_service=llmisvc)
    url = base_url + "/version"
    ca_cert = None if insecure else _resolve_ca_cert(llmisvc.client)

    LOGGER.info(f"Querying vLLM version from {llmisvc.name} at {url}")
    status_code, response_body = _curl_get(url=url, token=token, ca_cert=ca_cert)
    if status_code != 200:
        raise ValueError(f"vLLM /version returned {status_code}: {response_body}")

    try:
        data: dict[str, Any] = json.loads(response_body)
        version = data["version"]
        if not isinstance(version, str):
            raise TypeError(f"Expected version to be str, got {type(version).__name__}")
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ValueError(f"Failed to parse vLLM version response: {e}\nBody: {response_body[:500]}") from e

    LOGGER.info(f"vLLM version for {llmisvc.name}: {version}")
    return version


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
    base_url = get_llm_inference_url(llm_service=llmisvc)
    url = base_url + LLMEndpoint.CHAT_COMPLETIONS
    model_name = _get_model_name(llmisvc=llmisvc)
    body = _build_chat_body(model_name=model_name, prompt=prompt)
    ca_cert = None if insecure else _resolve_ca_cert(llmisvc.client)

    LOGGER.info(f"Sending inference request to {llmisvc.name} — URL: {url}, Model: {model_name}")
    status_code, response_body = _curl_post(url=url, body=body, token=token, ca_cert=ca_cert)
    LOGGER.info(f"Inference response — status={status_code}\n{response_body}")
    return status_code, response_body


def _build_completions_body(model_name: str, prompt: str, max_tokens: int = LLMEndpoint.DEFAULT_MAX_TOKENS) -> str:
    """Build OpenAI completions request body (non-chat)."""
    return json.dumps({
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": LLMEndpoint.DEFAULT_TEMPERATURE,
        "stream": False,
    })


def send_completions(
    llmisvc: LLMInferenceService,
    prompt: str,
    token: str | None = None,
    insecure: bool = True,
) -> tuple[int, str]:
    """Send a completions request to /v1/completions.

    Args:
        llmisvc: The LLMInferenceService to send the request to.
        prompt: The prompt text.
        token: Optional bearer token for authentication.
        insecure: Skip TLS verification (default True).

    Returns:
        Tuple of (status_code, response_body).
    """
    base_url = get_llm_inference_url(llm_service=llmisvc)
    url = base_url + LLMEndpoint.COMPLETIONS
    model_name = _get_model_name(llmisvc=llmisvc)
    body = _build_completions_body(model_name=model_name, prompt=prompt)
    ca_cert = None if insecure else _resolve_ca_cert(llmisvc.client)

    LOGGER.info(f"Sending completions request to {llmisvc.name} — URL: {url}, Model: {model_name}")
    status_code, response_body = _curl_post(url=url, body=body, token=token, ca_cert=ca_cert)
    LOGGER.info(f"Completions response — status={status_code}\n{response_body}")
    return status_code, response_body


def parse_completion_text(response_body: str) -> str:
    """Extract completion text from a chat completion response."""
    try:
        data = json.loads(response_body)
        return data["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Failed to parse completion response: {e}\nBody: {response_body[:500]}") from e


def parse_prompt_tokens(response_body: str) -> int:
    """Extract prompt_tokens from an inference response's usage field.

    Args:
        response_body: JSON response body from a chat completion or completion request.

    Returns:
        Number of prompt tokens reported by the model.
    """
    try:
        data = json.loads(response_body)
        return data["usage"]["prompt_tokens"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ValueError(f"Failed to parse prompt_tokens: {e}\nBody: {response_body[:500]}") from e


@retry(wait_timeout=120, sleep=10, exceptions_dict={AssertionError: []}, print_log=False)
def assert_kv_transfer(
    prometheus: Prometheus,
    unprivileged_client: DynamicClient,
    llmisvc: LLMInferenceService,
    expected_transferred_tokens: int,
    num_requests: int,
) -> bool:
    """Assert P/D KV transfer metrics match expected values.

    In Prefill/Decode disaggregation, the prefill engine computes all prompt tokens
    locally and transfers the KV cache to the decode engine via NIXL. The decode
    engine receives all tokens externally and reports 0 local compute.

    Note: the decode engine actually recomputes 1 token per request for the forward
    pass, but vLLM 0.21+ has a metrics bug where this is not reflected in the
    local_compute counter (the metric snapshot is taken before the recomputation).

    Expected metric values on kserve_vllm:prompt_tokens_by_source_total:
    - prefill pod, source=local_compute: all prompt tokens (prefill did the work)
    - prefill pod, source=external_kv_transfer: 0 (prefill never receives KV)
    - decode pod, source=external_kv_transfer: all prompt tokens (received from prefill)
    - decode pod, source=local_compute: 0 (see note above about vLLM metrics bug)

    Args:
        prometheus: Prometheus client for querying metrics.
        unprivileged_client: DynamicClient instance.
        llmisvc: The LLMInferenceService under test.
        expected_transferred_tokens: sum of prompt_tokens from all responses.
        num_requests: number of requests sent (unused, kept for future upstream fix).

    Returns:
        True when all assertions pass (required by @retry to stop retrying).
    """
    prompt_tokens_by_source_total_metric = "kserve_vllm:prompt_tokens_by_source_total"

    decode_pod = get_llmd_pod_by_role(client=unprivileged_client, llmisvc=llmisvc, role="decode")
    prefill_pod = get_llmd_pod_by_role(client=unprivileged_client, llmisvc=llmisvc, role="prefill")

    def _query(pod_name: str, source: str) -> float:
        query = (
            f"{prompt_tokens_by_source_total_metric}"
            f'{{namespace="{llmisvc.namespace}",pod="{pod_name}",source="{source}"}}'
        )
        raw = get_metrics_value(prometheus=prometheus, metrics_query=query)
        LOGGER.info(f"PromQL: {query} → {raw}")
        return float(raw or 0)

    decode_kv = _query(pod_name=decode_pod.name, source="external_kv_transfer")
    decode_local = _query(pod_name=decode_pod.name, source="local_compute")
    prefill_compute = _query(pod_name=prefill_pod.name, source="local_compute")
    prefill_kv = _query(pod_name=prefill_pod.name, source="external_kv_transfer")

    LOGGER.info(
        f"KV transfer metrics — "
        f"decode.external_kv_transfer={decode_kv}, "
        f"decode.local_compute={decode_local}, "
        f"prefill.local_compute={prefill_compute}, "
        f"prefill.external_kv_transfer={prefill_kv}, "
        f"expected_transferred_tokens={expected_transferred_tokens}"
    )

    assert decode_kv == expected_transferred_tokens, (
        f"decode.external_kv_transfer={decode_kv} != expected {expected_transferred_tokens}"
    )
    # In P/D mode the decode engine receives ALL prompt tokens via KV transfer from
    # the prefill engine. It then recomputes 1 token locally to run a forward pass
    # for sampling. However, vLLM 0.21+ has a metrics bug: the metric snapshot is
    # taken BEFORE that recomputation happens, so local_compute reports 0 instead
    # of 1 per request. The actual model behavior is correct — only the metric is wrong.
    assert decode_local == 0, (
        f"decode.local_compute={decode_local} != 0 (vLLM 0.21+ does not count the recomputed token in PD mode metrics)"
    )
    assert prefill_compute == expected_transferred_tokens, (
        f"prefill.local_compute={prefill_compute} != expected {expected_transferred_tokens}"
    )
    assert prefill_kv == 0, f"Prefill pod should not receive KV transfers (got {prefill_kv})"
    return True


def get_llmd_pod_by_role(
    client: DynamicClient,
    llmisvc: LLMInferenceService,
    role: str,
) -> Pod:
    """Get the workload pod with a specific llm-d.ai/role label.

    Args:
        client: DynamicClient instance.
        llmisvc: The LLMInferenceService to get the pod for.
        role: Pod role label value (decode or prefill).

    Returns:
        The matching Pod object.

    Raises:
        RuntimeError: If no pod with the given role is found.
    """
    for pod in Pod.get(
        client=client,
        namespace=llmisvc.namespace,
        label_selector=(
            f"{Pod.ApiGroup.APP_KUBERNETES_IO}/part-of=llminferenceservice,"
            f"{Pod.ApiGroup.APP_KUBERNETES_IO}/name={llmisvc.name},"
            f"llm-d.ai/role={role}"
        ),
    ):
        return pod
    raise RuntimeError(f"No pod with role={role} for {llmisvc.name} in {llmisvc.namespace}")


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


def scheduler_has_plugin(
    client: DynamicClient,
    llmisvc: LLMInferenceService,
    plugin_name: str,
) -> bool:
    """Check whether a named plugin exists in the scheduler's --config-text.

    Args:
        client: DynamicClient instance.
        llmisvc: The LLMInferenceService to check.
        plugin_name: Name of the plugin to search for (e.g. 'prefill-filter').

    Returns:
        True if the plugin is found in the scheduler config.

    Raises:
        RuntimeError: If no scheduler pod or --config-text is found.
    """
    pod = get_llmd_router_scheduler_pod(client=client, llmisvc=llmisvc)
    if not pod:
        raise RuntimeError(f"No scheduler pod found for {llmisvc.name} in {llmisvc.namespace}")

    containers = pod.instance.spec.containers
    for container in containers:
        args = container.get("args") or []
        for i, arg in enumerate(args):
            if arg == "--config-text" and i + 1 < len(args):
                return plugin_name in args[i + 1]

    raise RuntimeError(f"No --config-text found in scheduler pod {pod.name}")


@retry(wait_timeout=120, sleep=10, exceptions_dict={AssertionError: []}, print_log=False)
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
    count: int,
    max_failures: int = 5,
    delay_after_first_request: int | None = None,
) -> int:
    """Send identical chat completion requests until ``count`` succeed.

    Keeps sending the same prompt until the target number of successful (HTTP 200)
    responses is reached. Aborts with AssertionError if failures exceed ``max_failures``.

    Args:
        llmisvc: The LLMInferenceService to send requests to.
        prompt: The prompt text sent in every request.
        token: Bearer token for authentication.
        count: Number of successful responses required.
        max_failures: Maximum tolerated failures (non-200 or exceptions) before aborting.
        delay_after_first_request: Seconds to wait after the first successful request,
            used to allow KV cache index propagation before subsequent requests.

    Returns:
        The number of successful requests (always equal to ``count``).

    Raises:
        AssertionError: If failures exceed ``max_failures``.
    """
    LOGGER.info(f"Sending requests until {count} succeed (max {max_failures} failures allowed)")
    successful = 0
    failures = 0

    while successful < count:
        # mark test failed when inference requests exceed the max_failures threshold
        assert failures < max_failures, f"Too many failures: {failures}/{max_failures}, {successful}/{count} succeeded"

        try:
            status, body = send_chat_completions(llmisvc=llmisvc, prompt=prompt, token=token, insecure=False)
        except Exception:
            failures += 1
            LOGGER.exception(f"Request raised an exception ({failures}/{max_failures} failures)")
            continue

        if status == 200:
            successful += 1
            # add delay after first successful request for KV cache index propagation
            if successful == 1 and delay_after_first_request:
                LOGGER.info(f"Waiting {delay_after_first_request}s for KV cache index propagation")
                time.sleep(delay_after_first_request)
        else:
            failures += 1
            LOGGER.warning(f"Request failed with status {status}: {body} ({failures}/{max_failures} failures)")
            time.sleep(5)

    LOGGER.info(f"{successful} requests succeeded ({failures} failures)")
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


def workaround_503_no_healthy_upstream(llmisvc: LLMInferenceService, prompt: str) -> None:
    """Warm up inference endpoint to work around RHOAIENG-55154.

    Requests soon after Ready condition may 503 with 'no healthy upstream'.
    Retries every 3s for up to 30s until the endpoint stops returning 503.
    Swallows TimeoutExpiredError if retries are exhausted, letting the real test assertion decide.
    Skips entirely if the Jira issue is closed (result is cached).

    See: https://redhat.atlassian.net/browse/RHOAIENG-55154

    Args:
        llmisvc: The LLMInferenceService to warm up
        prompt: The prompt to send in the warm up request
    """
    if not is_jira_issue_open(jira_id="RHOAIENG-55154"):
        LOGGER.info("RHOAIENG-55154 is closed - remove this block")
        return

    try:
        _send_warm_up_request(llmisvc=llmisvc, prompt=prompt)
    except TimeoutExpiredError:
        LOGGER.warning(f"RHOAIENG-55154: warm up retries exhausted for {llmisvc.name}")


@retry(wait_timeout=30, sleep=3)
def _send_warm_up_request(llmisvc: LLMInferenceService, prompt: str) -> bool:
    """Send one warm-up request; return True to stop retrying, False to retry."""
    LOGGER.info(f"RHOAIENG-55154: sending warm up request to {llmisvc.name}")
    status, body = send_chat_completions(llmisvc=llmisvc, prompt=prompt)
    LOGGER.info(f"RHOAIENG-55154: warm up returned {status}")
    return not (status == 503 and "no healthy upstream" in body)
