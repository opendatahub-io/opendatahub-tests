"""
Tests for RBAC authorization violations on auth-protected KServe inference endpoints.

When a user has a valid token but lacks permission to access a specific model,
kube-rbac-proxy must return HTTP 403 Forbidden. This covers:
    - Valid token scoped to a different namespace (cross-namespace access)
    - ServiceAccount token without required inference RBAC roles

These are distinct from 401 scenarios: the identity IS valid, but authorization
is denied. Multi-tenant clusters require strict namespace isolation.
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


def _send_request_with_token(
    inference_service: InferenceService,
    token: str,
    body: str,
) -> tuple[int, str]:
    """Send an inference request with a specific bearer token."""
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
class TestRbacForbidden:
    """Auth-protected ISVC rejects tokens that lack model access with HTTP 403.

    Preconditions:
        - InferenceService deployed with enable_auth=True (kube-rbac-proxy)
        - A ServiceAccount in a different namespace with no RBAC for the model

    Expected Results:
        - HTTP 403 Forbidden for cross-namespace or unauthorized tokens
        - Pod remains healthy after authorization failures
    """

    def test_cross_namespace_token_returns_403(
        self,
        negative_test_ovms_isvc_with_auth: InferenceService,
        cross_namespace_sa_token: str,
    ) -> None:
        """Verify that a token from a different namespace returns 403.

        Given an auth-protected InferenceService in namespace A
        When sending an inference request with a token scoped to namespace B
        Then the response must be HTTP 403 Forbidden
        """
        status_code, response_body = _send_request_with_token(
            inference_service=negative_test_ovms_isvc_with_auth,
            token=cross_namespace_sa_token,
            body=_VALID_BODY,
        )

        assert status_code == HTTPStatus.FORBIDDEN, (
            f"Expected 403 for cross-namespace token, got {status_code}. Response: {response_body[:200]}"
        )

    def test_unauthorized_sa_token_returns_403(
        self,
        negative_test_ovms_isvc_with_auth: InferenceService,
        unauthorized_sa_token: str,
    ) -> None:
        """Verify that a SA token without inference roles returns 403.

        Given an auth-protected InferenceService
        When sending a request with a ServiceAccount token that has no inference RBAC
        Then the response must be HTTP 403 Forbidden
        """
        status_code, response_body = _send_request_with_token(
            inference_service=negative_test_ovms_isvc_with_auth,
            token=unauthorized_sa_token,
            body=_VALID_BODY,
        )

        assert status_code == HTTPStatus.FORBIDDEN, (
            f"Expected 403 for unauthorized SA token, got {status_code}. Response: {response_body[:200]}"
        )

    def test_pod_remains_healthy_after_forbidden_requests(
        self,
        admin_client: DynamicClient,
        negative_test_ovms_isvc_with_auth: InferenceService,
        cross_namespace_sa_token: str,
        initial_pod_state_auth: dict[str, dict[str, Any]],
    ) -> None:
        """Verify that pods remain healthy after repeated 403 responses.

        Given an auth-protected InferenceService
        When sending multiple requests with unauthorized tokens
        Then the pods must not restart or crash
        """
        for i in range(3):
            status_code, _ = _send_request_with_token(
                inference_service=negative_test_ovms_isvc_with_auth,
                token=cross_namespace_sa_token,
                body=_VALID_BODY,
            )
            assert status_code == HTTPStatus.FORBIDDEN, (
                f"Request {i + 1} expected 403 before pod health check, got {status_code}"
            )

        assert_pods_healthy(
            admin_client=admin_client,
            isvc=negative_test_ovms_isvc_with_auth,
            initial_pod_state=initial_pod_state_auth,
        )
