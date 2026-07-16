"""
Tests for authentication failure handling on auth-protected KServe inference endpoints.

When KServe raw deployment auth is enabled (kube-rbac-proxy sidecar), requests
without a valid bearer token must be rejected with HTTP 401 Unauthorized. This is
the most common customer misconfiguration when first enabling auth on model serving.

Test scenarios:
    - No Authorization header at all
    - Malformed token (random string, not a valid SA token)
    - Invalid JWT structure
    - Wrong auth scheme

Security validation:
    - Error responses must NOT leak internal paths, token values, or pod names
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

_INVALID_TOKENS: list[tuple[str, str]] = [
    ("", "no_token"),
    ("Bearer totally-not-a-real-token", "random_string_token"),
    ("Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.expired.signature", "malformed_jwt"),
    ("InvalidScheme somevalue", "wrong_auth_scheme"),
]


def _send_request_with_auth(
    inference_service: InferenceService,
    auth_header: str,
    body: str,
) -> tuple[int, str]:
    """Send an inference request with a specific Authorization header value."""
    base_url = inference_service.instance.status.url
    if not base_url:
        raise ValueError(f"InferenceService '{inference_service.name}' has no URL; is it Ready?")

    endpoint = f"{base_url}/v2/models/{inference_service.name}/infer"

    auth_flag = f"-H 'Authorization: {auth_header}'" if auth_header else ""
    cmd = (
        f"curl -s -w '\\n%{{http_code}}' "
        f"-X POST {endpoint} "
        f"-H 'Content-Type: application/json' "
        f"{auth_flag} "
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
class TestAuthUnauthorized:
    """Auth-protected ISVC rejects requests without valid authentication with HTTP 401.

    Preconditions:
        - InferenceService deployed with enable_auth=True (kube-rbac-proxy)
        - ISVC is Ready and serving

    Expected Results:
        - HTTP 401 Unauthorized for all invalid/missing token scenarios
        - Error response does not leak sensitive information
        - Pod remains healthy after auth failures
    """

    @pytest.mark.parametrize(
        ("auth_header", "case_id"),
        [pytest.param(token, case_id, id=case_id) for token, case_id in _INVALID_TOKENS],
    )
    def test_invalid_token_returns_401(
        self,
        negative_test_ovms_isvc_with_auth: InferenceService,
        auth_header: str,
        case_id: str,
    ) -> None:
        """Verify that invalid/missing auth tokens return HTTP 401.

        Given an auth-protected InferenceService is deployed and ready
        When sending inference requests with invalid/missing Authorization headers
        Then the response must be HTTP 401 Unauthorized
        """
        status_code, response_body = _send_request_with_auth(
            inference_service=negative_test_ovms_isvc_with_auth,
            auth_header=auth_header,
            body=_VALID_BODY,
        )

        assert status_code == HTTPStatus.UNAUTHORIZED, (
            f"[{case_id}] Expected 401 for invalid auth, got {status_code}. Response: {response_body[:200]}"
        )

    def test_error_response_does_not_leak_sensitive_info(
        self,
        negative_test_ovms_isvc_with_auth: InferenceService,
    ) -> None:
        """Verify that auth error responses do not contain internal paths or tokens.

        Given an auth-protected InferenceService
        When sending a request without authentication
        Then the error response body must not contain pod names, internal paths, or tokens
        """
        _, response_body = _send_request_with_auth(
            inference_service=negative_test_ovms_isvc_with_auth,
            auth_header="",
            body=_VALID_BODY,
        )

        sensitive_patterns = [
            "/var/run/",
            "/etc/",
            "kube-system",
            "serviceaccount",
            "redhat-ods-applications",
        ]
        for pattern in sensitive_patterns:
            assert pattern not in response_body, (
                f"Auth error response leaks sensitive info: found '{pattern}' in: {response_body[:500]}"
            )

    def test_pod_remains_healthy_after_auth_failures(
        self,
        admin_client: DynamicClient,
        negative_test_ovms_isvc_with_auth: InferenceService,
        initial_pod_state_auth: dict[str, dict[str, Any]],
    ) -> None:
        """Verify that predictor pods remain healthy after repeated auth failures.

        Given an auth-protected InferenceService
        When sending multiple requests with invalid auth
        Then the pods must not restart or crash
        """
        for auth_header, case_id in _INVALID_TOKENS:
            status_code, _ = _send_request_with_auth(
                inference_service=negative_test_ovms_isvc_with_auth,
                auth_header=auth_header,
                body=_VALID_BODY,
            )
            assert status_code == HTTPStatus.UNAUTHORIZED, (
                f"[{case_id}] Expected 401 before pod health check, got {status_code}"
            )

        assert_pods_healthy(
            admin_client=admin_client,
            isvc=negative_test_ovms_isvc_with_auth,
            initial_pod_state=initial_pod_state_auth,
        )
