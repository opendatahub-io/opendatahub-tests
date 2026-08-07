"""
Tests for request overload / queue overflow on single-replica ISVC.

When a single-replica InferenceService is flooded with more concurrent valid
requests than it can process, the server must gracefully reject excess requests
with 429 (Too Many Requests) or 503 (Service Unavailable) rather than crashing
the pod or silently dropping requests.

Customer scenario:
    A production traffic spike sends 50+ simultaneous inference requests to an
    ISVC with 1 replica. Without proper queue management, the pod OOMs or the
    proxy drops connections silently.
"""

import json
from http import HTTPStatus
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from pytest_testconfig import config as py_config

from tests.model_serving.model_server.kserve.negative.utils import (
    VALID_OVMS_INFERENCE_BODY,
    assert_kserve_control_plane_stable,
    assert_pods_healthy,
    send_inference_requests_concurrently,
    snapshot_kserve_control_plane_restart_totals,
)

pytestmark = [
    pytest.mark.rawdeployment,
    pytest.mark.usefixtures("valid_aws_config"),
]

_VALID_BODY: str = json.dumps(VALID_OVMS_INFERENCE_BODY)

OVERLOAD_REQUEST_COUNT: int = 30

OVERLOAD_ACCEPTABLE_CODES: set[int] = {
    HTTPStatus.OK,  # 200 - some requests may succeed
    HTTPStatus.TOO_MANY_REQUESTS,  # 429 - rate limiting
    HTTPStatus.SERVICE_UNAVAILABLE,  # 503 - temporary overload
    HTTPStatus.BAD_GATEWAY,  # 502 - upstream exhausted
    HTTPStatus.GATEWAY_TIMEOUT,  # 504 - proxy timeout
}


@pytest.mark.tier3
@pytest.mark.rawdeployment
class TestRequestOverload:
    """Single-replica ISVC gracefully handles concurrent request overload.

    Preconditions:
        - OVMS RawDeployment InferenceService with 1 replica deployed and Ready
        - Valid ONNX inference body available

    Expected Results:
        - All responses have acceptable status codes (200/429/503/502/504)
        - No 5xx responses that indicate server crashes (500)
        - Pod remains running with no crashes after overload
        - Control plane remains stable
    """

    def test_overload_produces_acceptable_responses(
        self,
        negative_test_ovms_isvc: InferenceService,
    ) -> None:
        """Verify that overloading a single-replica ISVC produces graceful responses.

        Given a single-replica InferenceService
        When sending 30 concurrent valid inference requests
        Then all responses must have acceptable status codes (no 500 Internal Server Error)
        """
        results = send_inference_requests_concurrently(
            inference_service=negative_test_ovms_isvc,
            body=_VALID_BODY,
            count=OVERLOAD_REQUEST_COUNT,
        )

        assert len(results) == OVERLOAD_REQUEST_COUNT, f"Expected {OVERLOAD_REQUEST_COUNT} results, got {len(results)}"

        for idx, (status_code, response_body) in enumerate(results):
            assert status_code in OVERLOAD_ACCEPTABLE_CODES, (
                f"Request #{idx + 1}: got unexpected {status_code} during overload. Response: {response_body[:200]}"
            )
            assert status_code != HTTPStatus.INTERNAL_SERVER_ERROR, (
                f"Request #{idx + 1}: server returned 500 during overload - indicates crash. "
                f"Response: {response_body[:200]}"
            )

    def test_pod_survives_overload(
        self,
        admin_client: DynamicClient,
        negative_test_ovms_isvc: InferenceService,
        initial_pod_state: dict[str, dict[str, Any]],
    ) -> None:
        """Verify that the predictor pod survives a request overload burst.

        Given a single-replica InferenceService
        When flooding it with 30 concurrent valid requests
        Then the pod must remain Running with no new restarts
        """
        send_inference_requests_concurrently(
            inference_service=negative_test_ovms_isvc,
            body=_VALID_BODY,
            count=OVERLOAD_REQUEST_COUNT,
        )

        assert_pods_healthy(
            admin_client=admin_client,
            isvc=negative_test_ovms_isvc,
            initial_pod_state=initial_pod_state,
        )

    def test_control_plane_stable_after_overload(
        self,
        admin_client: DynamicClient,
        negative_test_ovms_isvc: InferenceService,
    ) -> None:
        """Verify KServe control plane is unaffected by data-plane overload.

        Given a single-replica InferenceService
        When the data plane is flooded with concurrent requests
        Then the control plane must remain Available with no new restarts
        """
        applications_namespace: str = py_config["applications_namespace"]
        prior_totals = snapshot_kserve_control_plane_restart_totals(
            admin_client=admin_client,
            applications_namespace=applications_namespace,
        )

        send_inference_requests_concurrently(
            inference_service=negative_test_ovms_isvc,
            body=_VALID_BODY,
            count=OVERLOAD_REQUEST_COUNT,
        )

        assert_kserve_control_plane_stable(
            admin_client=admin_client,
            applications_namespace=applications_namespace,
            prior_restart_totals=prior_totals,
        )

    def test_inference_recovers_after_overload(
        self,
        negative_test_ovms_isvc: InferenceService,
    ) -> None:
        """Verify that normal inference succeeds after an overload burst clears.

        Given a single-replica InferenceService that was just overloaded
        When sending a single valid request after the burst
        Then the response must be HTTP 200 with valid outputs
        """
        send_inference_requests_concurrently(
            inference_service=negative_test_ovms_isvc,
            body=_VALID_BODY,
            count=OVERLOAD_REQUEST_COUNT,
        )

        from tests.model_serving.model_server.kserve.negative.utils import send_inference_request

        status_code, response_body = send_inference_request(
            inference_service=negative_test_ovms_isvc,
            body=_VALID_BODY,
        )

        assert status_code == HTTPStatus.OK, (
            f"Post-overload recovery request returned {status_code}. Response: {response_body[:200]}"
        )
        parsed = json.loads(response_body)
        assert parsed.get("outputs"), f"Post-overload recovery returned empty outputs. Response: {response_body[:200]}"
