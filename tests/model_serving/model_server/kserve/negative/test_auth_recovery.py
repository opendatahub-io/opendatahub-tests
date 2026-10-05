"""
Tests verifying inference recovery after authentication failures.

After receiving multiple requests with invalid/missing tokens, the auth-protected
ISVC must still accept and process a subsequent request with a valid token.
This proves that auth failures do not corrupt endpoint state or leave the
kube-rbac-proxy sidecar in a broken state.

Customer scenario:
    A developer misconfigures their client token, gets repeated 401s, then
    fixes the token. The model must respond immediately without needing
    a pod restart or ISVC recreation.
"""

import json
import shlex
from http import HTTPStatus

import pytest
from ocp_resources.inference_service import InferenceService
from pyhelper_utils.shell import run_command

from tests.model_serving.model_server.kserve.negative.utils import (
    VALID_OVMS_INFERENCE_BODY,
)

pytestmark = [
    pytest.mark.rawdeployment,
    pytest.mark.usefixtures("valid_aws_config"),
]

_VALID_BODY: str = json.dumps(VALID_OVMS_INFERENCE_BODY)


def _send_authed_request(
    inference_service: InferenceService,
    token: str,
    body: str,
) -> tuple[int, str]:
    """Send an inference request with a bearer token."""
    base_url = inference_service.instance.status.url
    if not base_url:
        raise ValueError(f"InferenceService '{inference_service.name}' has no URL; is it Ready?")

    endpoint = f"{base_url}/v2/models/{inference_service.name}/infer"

    cmd = (
        f"curl -s -w '\\n%{{http_code}}' "
        f"-X POST {endpoint} "
        f"-H 'Content-Type: application/json' "
        f"-H 'Authorization: Bearer {token}' "
        f"--data-raw {shlex.quote(body)} "
        f"--insecure"
    )

    _, out, _ = run_command(command=shlex.split(cmd), verify_stderr=False, check=False)

    lines = out.strip().split("\n")
    try:
        status_code = int(lines[-1])
    except ValueError as exc:
        raise ValueError(f"Could not parse HTTP status code from curl output: {out!r}") from exc
    return status_code, "\n".join(lines[:-1])


@pytest.mark.tier2
class TestAuthRecovery:
    """Valid inference succeeds immediately after repeated auth failures.

    Preconditions:
        - InferenceService deployed with enable_auth=True and Ready
        - A valid bearer token (SA with RBAC) that grants access to the model

    Expected Results:
        - After sending 5 requests with invalid tokens (all return 401)
        - A subsequent request with a valid token returns HTTP 200
        - Response body contains non-empty outputs
    """

    def test_valid_request_succeeds_after_auth_failures(
        self,
        negative_test_ovms_isvc_with_auth: InferenceService,
        valid_inference_token: str,
    ) -> None:
        """Verify that valid inference works immediately after auth failures.

        Given an auth-protected InferenceService
        When sending 5 requests with invalid tokens (expect 401)
        And then sending 1 request with a valid token
        Then the valid request must return HTTP 200 with inference outputs
        """
        for _ in range(5):
            status_code, _ = _send_authed_request(
                inference_service=negative_test_ovms_isvc_with_auth,
                token="invalid-token-that-should-fail",
                body=_VALID_BODY,
            )
            assert status_code == HTTPStatus.UNAUTHORIZED, f"Expected 401 for invalid token, got {status_code}"

        status_code, response_body = _send_authed_request(
            inference_service=negative_test_ovms_isvc_with_auth,
            token=valid_inference_token,
            body=_VALID_BODY,
        )

        assert status_code == HTTPStatus.OK, (
            f"Valid request after auth failures returned {status_code}. Response: {response_body[:200]}"
        )
        parsed = json.loads(response_body)
        assert parsed.get("outputs"), (
            f"Valid request returned empty outputs after auth failures. Response: {response_body[:200]}"
        )
