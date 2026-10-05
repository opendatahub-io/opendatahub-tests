"""
Tests for oversized payload handling in KServe inference requests.

Sending a request body that far exceeds typical server buffer limits (4-8 MB)
exercises the boundary between "large but valid" and "reject as too large".
KServe / envoy should return a 4xx (commonly 413 Request Entity Too Large)
and must not crash or restart the predictor pod.

Boundary condition:
    ``OVERSIZED_PAYLOAD_SIZE_BYTES`` is set to 6 MB in constants, which exceeds
    the default envoy per-request body limit of 4 MB used in many RHOAI deployments.
    Servers that have a higher or unlimited buffer may return 400 (bad request)
    because the body is not valid JSON, which is also acceptable.
"""

import shlex
import tempfile
from http import HTTPStatus
from pathlib import Path
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from pyhelper_utils.shell import run_command

from tests.model_serving.model_server.kserve.negative.constants import OVERSIZED_PAYLOAD_SIZE_BYTES
from tests.model_serving.model_server.kserve.negative.utils import (
    assert_pods_healthy,
)

pytestmark = pytest.mark.usefixtures("valid_aws_config")

OVERSIZED_PAYLOAD_EXPECTED_CODES: set[int] = {
    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
    HTTPStatus.BAD_REQUEST,
    HTTPStatus.PRECONDITION_FAILED,
    HTTPStatus.REQUEST_TIMEOUT,
    HTTPStatus.SERVICE_UNAVAILABLE,
}


def _send_oversized_request(inference_service: InferenceService) -> tuple[int, str]:
    """Send a 6 MB payload via curl stdin to avoid E2BIG on argv."""
    base_url = inference_service.instance.status.url
    if not base_url:
        raise ValueError(f"InferenceService '{inference_service.name}' has no URL; is it Ready?")

    endpoint = f"{base_url}/v2/models/{inference_service.name}/infer"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".bin", delete=False) as tmp:
        tmp.write("A" * OVERSIZED_PAYLOAD_SIZE_BYTES)
        tmp_path = tmp.name

    try:
        cmd = (
            f"curl -s -w '\\n%{{http_code}}' "
            f"-X POST {endpoint} "
            f"-H 'Content-Type: application/json' "
            f"--data-binary @{tmp_path} "
            f"--insecure"
        )
        _, out, _ = run_command(command=shlex.split(cmd), verify_stderr=False, check=False)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    lines = out.strip().split("\n")
    try:
        status_code = int(lines[-1])
    except ValueError as exc:
        raise ValueError(f"Could not parse HTTP status code from curl output: {out!r}") from exc
    return status_code, "\n".join(lines[:-1])


@pytest.mark.tier3
@pytest.mark.rawdeployment
class TestOversizedPayload:
    """KServe rejects or gracefully handles oversized inference request bodies.

    Preconditions:
        - OVMS RawDeployment InferenceService is deployed and Ready

    Expected Results:
        - HTTP Status Code: 413 or other 4xx / 503 indicating rejection
        - Predictor pod remains running with no new restarts
    """

    def test_oversized_payload_returns_error(
        self,
        negative_test_ovms_isvc: InferenceService,
    ) -> None:
        """Verify that sending a 6 MB body returns an error status code.

        Given an InferenceService is deployed and ready
        When sending a POST request with a 6 MB body that exceeds server limits
        Then the response must be a 4xx/503 error code indicating rejection
        """
        status_code, response_body = _send_oversized_request(inference_service=negative_test_ovms_isvc)

        assert status_code in OVERSIZED_PAYLOAD_EXPECTED_CODES, (
            f"Expected 413/400/408/503 for oversized payload ({OVERSIZED_PAYLOAD_SIZE_BYTES} bytes), "
            f"got {status_code}. Response (first 200 chars): {response_body[:200]}"
        )

    def test_model_pod_remains_healthy_after_oversized_payload(
        self,
        admin_client: DynamicClient,
        negative_test_ovms_isvc: InferenceService,
        initial_pod_state: dict[str, dict[str, Any]],
    ) -> None:
        """Verify that the predictor pod does not crash after an oversized payload.

        Given an InferenceService is deployed and ready
        When sending a 6 MB request body that should be rejected
        Then the same pods (by UID) should still be running without additional restarts
        """
        _send_oversized_request(inference_service=negative_test_ovms_isvc)
        assert_pods_healthy(
            admin_client=admin_client,
            isvc=negative_test_ovms_isvc,
            initial_pod_state=initial_pod_state,
        )
