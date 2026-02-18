"""Utility functions for negative inference tests."""

import json
import shlex
from typing import Any

from ocp_resources.inference_service import InferenceService
from pyhelper_utils.shell import run_command


def send_inference_request_with_content_type(
    inference_service: InferenceService,
    content_type: str,
    body: dict[str, Any],
) -> tuple[int, str]:
    """Send an inference request with a specific Content-Type header.

    This function is used for negative testing to verify error handling
    when sending requests with unsupported Content-Type headers.

    Args:
        inference_service: The InferenceService to send the request to.
        content_type: The Content-Type header value to use.
        body: The request body to send.

    Returns:
        A tuple of (status_code, response_body).

    Raises:
        ValueError: If the InferenceService has no URL or curl output is malformed.
    """
    url = inference_service.instance.status.url
    if not url:
        raise ValueError(f"InferenceService '{inference_service.name}' has no URL; is it Ready?")

    endpoint = f"{url}/v2/models/{inference_service.name}/infer"

    cmd = (
        f"curl -s -w '\\n%{{http_code}}' "
        f"-X POST {endpoint} "
        f"-H 'Content-Type: {content_type}' "
        f"-d '{json.dumps(body)}' "
        f"--insecure"
    )

    _, out, _ = run_command(command=shlex.split(cmd), verify_stderr=False, check=False)

    lines = out.strip().split("\n")
    try:
        status_code = int(lines[-1])
    except ValueError as exc:
        raise ValueError(f"Could not parse HTTP status code from curl output: {out!r}") from exc

    response_body = "\n".join(lines[:-1])

    return status_code, response_body
