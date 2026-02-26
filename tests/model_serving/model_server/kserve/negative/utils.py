"""Utility functions for negative inference tests."""

import json
import shlex
from typing import Any

from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from pyhelper_utils.shell import run_command

from utilities.infra import get_pods_by_isvc_label

VALID_OVMS_INFERENCE_BODY: dict[str, Any] = {
    "inputs": [
        {
            "name": "Input3",
            "shape": [1, 1, 28, 28],
            "datatype": "FP32",
            "data": [0.0] * 784,
        }
    ]
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


def _get_isvc_base_url(inference_service: InferenceService) -> str:
    url = inference_service.instance.status.url
    if not url:
        raise ValueError(f"InferenceService '{inference_service.name}' has no URL; is it Ready?")
    return url


def _parse_curl_output(output: str) -> tuple[int, str]:
    lines = output.strip().split("\n")
    try:
        status_code = int(lines[-1])
    except ValueError as exc:
        raise ValueError(f"Could not parse HTTP status code from curl output: {output!r}") from exc
    return status_code, "\n".join(lines[:-1])


def send_inference_request_with_content_type(
    inference_service: InferenceService,
    content_type: str,
    body: dict[str, Any],
) -> tuple[int, str]:
    """Send an inference request with a specific Content-Type header.

    Args:
        inference_service: The InferenceService to send the request to.
        content_type: The Content-Type header value to use.
        body: The request body to send.

    Returns:
        A tuple of (status_code, response_body).

    Raises:
        ValueError: If the InferenceService has no URL or curl output is malformed.
    """
    base_url = _get_isvc_base_url(inference_service=inference_service)
    endpoint = f"{base_url}/v2/models/{inference_service.name}/infer"

    cmd = (
        f"curl -s -w '\\n%{{http_code}}' "
        f"-X POST {endpoint} "
        f"-H 'Content-Type: {content_type}' "
        f"-d '{json.dumps(body)}' "
        f"--insecure"
    )

    _, out, _ = run_command(command=shlex.split(cmd), verify_stderr=False, check=False)
    return _parse_curl_output(output=out)


def send_raw_inference_request(
    inference_service: InferenceService,
    raw_body: str,
    model_name: str | None = None,
) -> tuple[int, str]:
    """Send an inference request with a raw string body.

    Useful for testing malformed JSON, missing fields, or wrong data types
    where a dict body would be auto-serialized correctly.

    Args:
        inference_service: The InferenceService to send the request to.
        raw_body: The raw string payload to send (can be invalid JSON).
        model_name: Override the model name in the URL path.
            Defaults to the InferenceService name.

    Returns:
        A tuple of (status_code, response_body).

    Raises:
        ValueError: If the InferenceService has no URL or curl output is malformed.
    """
    base_url = _get_isvc_base_url(inference_service=inference_service)
    target_model = model_name or inference_service.name
    endpoint = f"{base_url}/v2/models/{target_model}/infer"

    cmd = (
        f"curl -s -w '\\n%{{http_code}}' "
        f"-X POST {endpoint} "
        f"-H 'Content-Type: application/json' "
        f"--data-raw {shlex.quote(raw_body)} "
        f"--insecure"
    )

    _, out, _ = run_command(command=shlex.split(cmd), verify_stderr=False, check=False)
    return _parse_curl_output(output=out)
