"""
Tests for wrong HTTP method handling on the KServe v2 inference endpoint.

The KServe v2 inference protocol mandates POST for ``/v2/models/{name}/infer``.
Sending GET, PUT, DELETE, or PATCH must be rejected with a client error (4xx)
and must not crash or restart the serving pod.

Edge / boundary cases covered:
    - GET (a common mistake when copy-pasting from health check examples)
    - PUT (a common mistake when expecting REST CRUD semantics)
    - DELETE (should never mutate serving state)
    - PATCH (partial-update attempt)
"""

import json
from http import HTTPStatus
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_server.kserve.negative.utils import (
    VALID_OVMS_INFERENCE_BODY,
    assert_pods_healthy,
    send_inference_request_with_method,
)

pytestmark = pytest.mark.usefixtures("valid_aws_config")

VALID_BODY_RAW: str = json.dumps(VALID_OVMS_INFERENCE_BODY)

# OVMS / Istio / envoy may return 404 or 405 for wrong methods depending on the
# proxy configuration – both are acceptable error responses.
WRONG_METHOD_EXPECTED_CODES: set[int] = {
    HTTPStatus.METHOD_NOT_ALLOWED,  # 405 - canonical "wrong method"
    HTTPStatus.NOT_ACCEPTABLE,  # 406 - OVMS returns this with "Unsupported method"
    HTTPStatus.NOT_FOUND,  # 404 - some proxies return this when route is method-scoped
    HTTPStatus.BAD_REQUEST,  # 400 - permissive proxies that inspect body first
}


@pytest.mark.tier3
@pytest.mark.rawdeployment
class TestWrongHttpMethod:
    """KServe inference endpoint rejects non-POST methods with a 4xx status code.

    Preconditions:
        - OVMS RawDeployment InferenceService is deployed and Ready
        - A valid JSON inference body is available for reuse

    Expected Results:
        - HTTP Status Code: 405 Method Not Allowed (or 404/400 from the proxy layer)
        - No pod crash or restart after any of the wrong-method requests
    """

    @pytest.mark.parametrize(
        "http_method",
        [
            pytest.param("GET", id="get_method"),
            pytest.param("PUT", id="put_method"),
            pytest.param("DELETE", id="delete_method"),
            pytest.param("PATCH", id="patch_method"),
        ],
    )
    def test_wrong_http_method_returns_4xx(
        self,
        negative_test_ovms_isvc: InferenceService,
        http_method: str,
    ) -> None:
        """Verify that using a non-POST method on the infer endpoint returns 4xx.

        Given an InferenceService is deployed and ready
        When sending a request to ``/v2/models/{name}/infer`` with an unsupported HTTP method
        Then the response must be a 4xx client error and the service must not crash
        """
        status_code, response_body = send_inference_request_with_method(
            inference_service=negative_test_ovms_isvc,
            http_method=http_method,
            body=VALID_BODY_RAW,
        )

        assert status_code in WRONG_METHOD_EXPECTED_CODES, (
            f"Expected a 4xx client error for HTTP {http_method}, got {status_code}. Response: {response_body}"
        )

    def test_model_pod_remains_healthy_after_wrong_method_requests(
        self,
        admin_client: DynamicClient,
        negative_test_ovms_isvc: InferenceService,
        initial_pod_state: dict[str, dict[str, Any]],
    ) -> None:
        """Verify that the model pod remains healthy after wrong-method requests.

        Given an InferenceService is deployed and ready
        When sending GET and DELETE requests to the inference endpoint
        Then the same pods (by UID) should still be running without additional restarts
        """
        for method in ("GET", "DELETE"):
            send_inference_request_with_method(
                inference_service=negative_test_ovms_isvc,
                http_method=method,
                body=VALID_BODY_RAW,
            )
        assert_pods_healthy(
            admin_client=admin_client,
            isvc=negative_test_ovms_isvc,
            initial_pod_state=initial_pod_state,
        )
