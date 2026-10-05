"""
Tests for inference request timeout handling.

When a model takes too long to respond (or the connection to the model server
times out), the client must receive a clear timeout error (408/504) rather than
hanging indefinitely. This is the most common production complaint for large
model inference where gateway timeouts are misconfigured.

Test approach:
    Send a request with an artificially short client-side timeout (--max-time)
    to simulate what happens when the gateway or client timeout fires before
    the model responds. The ISVC itself is healthy; we are testing the timeout
    path, not a broken model.
"""

import json
import shlex
from http import HTTPStatus
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from pyhelper_utils.shell import run_command

from tests.model_serving.model_server.kserve.negative.utils import (
    VALID_OVMS_INFERENCE_BODY,
    assert_pods_healthy,
)

pytestmark = [
    pytest.mark.rawdeployment,
    pytest.mark.usefixtures("valid_aws_config"),
]

_VALID_BODY: str = json.dumps(VALID_OVMS_INFERENCE_BODY)

TIMEOUT_EXPECTED_CODES: set[int] = {
    HTTPStatus.REQUEST_TIMEOUT,  # 408
    HTTPStatus.GATEWAY_TIMEOUT,  # 504
    HTTPStatus.SERVICE_UNAVAILABLE,  # 503
    0,  # curl exit code 28 (timeout) maps to status_code 0 in our parser
}


def _send_request_with_timeout(
    inference_service: InferenceService,
    timeout_seconds: float,
    body: str,
) -> tuple[int, str]:
    """Send inference request with a client-side max-time constraint."""
    base_url = inference_service.instance.status.url
    if not base_url:
        raise ValueError(f"InferenceService '{inference_service.name}' has no URL; is it Ready?")

    endpoint = f"{base_url}/v2/models/{inference_service.name}/infer"

    cmd = (
        f"curl -s -w '\\n%{{http_code}}' "
        f"-X POST {endpoint} "
        f"-H 'Content-Type: application/json' "
        f"--data-raw {shlex.quote(body)} "
        f"--max-time {timeout_seconds} "
        f"--insecure"
    )

    _, out, _ = run_command(command=shlex.split(cmd), verify_stderr=False, check=False)

    lines = out.strip().split("\n")
    if not lines or not lines[-1].strip():
        return 0, out.strip()
    try:
        status_code = int(lines[-1])
    except ValueError:
        return 0, out.strip()
    return status_code, "\n".join(lines[:-1])


@pytest.mark.tier3
@pytest.mark.rawdeployment
class TestInferenceTimeout:
    """Client-side timeout produces a clear error without crashing the server.

    Preconditions:
        - OVMS RawDeployment InferenceService deployed and Ready
        - Model responds successfully for normal requests

    Expected Results:
        - With very short timeout (0.001s): curl timeout (exit 28) or 408/504
        - Pod remains healthy after timeout events
        - Subsequent normal request (no timeout) succeeds with HTTP 200
    """

    def test_short_timeout_returns_timeout_error(
        self,
        negative_test_ovms_isvc: InferenceService,
    ) -> None:
        """Verify that an extremely short client timeout produces a timeout error.

        Given an InferenceService is deployed and ready
        When sending a request with --max-time 0.001 (1ms timeout)
        Then the response should indicate a timeout (408, 504, or curl exit 28)
        """
        status_code, response_body = _send_request_with_timeout(
            inference_service=negative_test_ovms_isvc,
            timeout_seconds=0.001,
            body=_VALID_BODY,
        )

        assert status_code in TIMEOUT_EXPECTED_CODES, (
            f"Expected timeout error (0/408/504/503), got {status_code}. Response: {response_body[:200]}"
        )

    def test_pod_remains_healthy_after_timeout(
        self,
        admin_client: DynamicClient,
        negative_test_ovms_isvc: InferenceService,
        initial_pod_state: dict[str, dict[str, Any]],
    ) -> None:
        """Verify that the predictor pod remains healthy after timeout events.

        Given an InferenceService is deployed and ready
        When sending multiple requests that timeout on the client side
        Then the pods must not restart or crash
        """
        for i in range(3):
            status_code, _ = _send_request_with_timeout(
                inference_service=negative_test_ovms_isvc,
                timeout_seconds=0.001,
                body=_VALID_BODY,
            )
            assert status_code in TIMEOUT_EXPECTED_CODES, (
                f"Request {i + 1} expected timeout (0/408/504/503), got {status_code}"
            )

        assert_pods_healthy(
            admin_client=admin_client,
            isvc=negative_test_ovms_isvc,
            initial_pod_state=initial_pod_state,
        )

    def test_normal_request_succeeds_after_timeouts(
        self,
        negative_test_ovms_isvc: InferenceService,
    ) -> None:
        """Verify that normal inference succeeds after timeout events.

        Given an InferenceService that has experienced client-side timeouts
        When sending a subsequent request with no timeout constraint
        Then the response must be HTTP 200 with valid outputs
        """
        for i in range(3):
            status_code, _ = _send_request_with_timeout(
                inference_service=negative_test_ovms_isvc,
                timeout_seconds=0.001,
                body=_VALID_BODY,
            )
            assert status_code in TIMEOUT_EXPECTED_CODES, f"Setup request {i + 1} expected timeout, got {status_code}"

        from tests.model_serving.model_server.kserve.negative.utils import send_inference_request

        status_code, response_body = send_inference_request(
            inference_service=negative_test_ovms_isvc,
            body=_VALID_BODY,
        )

        assert status_code == HTTPStatus.OK, (
            f"Normal request after timeouts returned {status_code}. Response: {response_body[:200]}"
        )
        parsed = json.loads(response_body)
        assert parsed.get("outputs"), (
            f"Normal request returned empty outputs after timeout events. Response: {response_body[:200]}"
        )
