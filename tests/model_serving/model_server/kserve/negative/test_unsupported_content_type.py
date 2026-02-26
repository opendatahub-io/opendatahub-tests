"""Tests for unsupported Content-Type headers in inference requests.

Jira: RHOAIENG-48283
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
    send_inference_request,
)

pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.tier1
@pytest.mark.rawdeployment
class TestUnsupportedContentType:
    """Test class for verifying error handling when using unsupported Content-Type headers.

    Preconditions:
        - InferenceService deployed and ready
        - Model accepts application/json content type

    Test Steps:
        1. Create InferenceService with OVMS runtime
        2. Wait for InferenceService status = Ready
        3. Send POST to inference endpoint with header Content-Type: text/xml
        4. Send POST with header Content-Type: application/x-www-form-urlencoded
        5. Capture responses for both requests
        6. Verify model pod health status

    Expected Results:
        - HTTP Status Code: 415 Unsupported Media Type for invalid Content-Types
        - Error indicates expected content type is application/json
        - Model pod remains healthy (Running, no restarts)
    """

    @pytest.mark.parametrize(
        "content_type",
        [
            pytest.param("text/xml", id="text_xml"),
            pytest.param("application/x-www-form-urlencoded", id="form_urlencoded"),
        ],
    )
    def test_unsupported_content_type_returns_415(
        self,
        negative_test_ovms_isvc: InferenceService,
        content_type: str,
    ) -> None:
        """Verify that unsupported Content-Type headers return 415 status code.

        Given an InferenceService is deployed and ready
        When sending a POST request with an unsupported Content-Type header
        Then the response should have HTTP status code 415 (Unsupported Media Type)
        """
        status_code, response_body = send_inference_request(
            inference_service=negative_test_ovms_isvc,
            body=json.dumps(VALID_OVMS_INFERENCE_BODY),
            content_type=content_type,
        )

        assert status_code == HTTPStatus.UNSUPPORTED_MEDIA_TYPE, (
            f"Expected 415 Unsupported Media Type for Content-Type '{content_type}', "
            f"got {status_code}. Response: {response_body}"
        )

    def test_model_pod_remains_healthy_after_invalid_requests(
        self,
        admin_client: DynamicClient,
        negative_test_ovms_isvc: InferenceService,
        initial_pod_state: dict[str, dict[str, Any]],
    ) -> None:
        """Verify that the model pod remains healthy after receiving invalid requests.

        Given an InferenceService is deployed and ready
        When sending requests with unsupported Content-Type headers
        Then the same pods (by UID) should still be running without additional restarts
        """
        send_inference_request(
            inference_service=negative_test_ovms_isvc,
            body=json.dumps(VALID_OVMS_INFERENCE_BODY),
            content_type="text/xml",
        )
        assert_pods_healthy(
            admin_client=admin_client,
            isvc=negative_test_ovms_isvc,
            initial_pod_state=initial_pod_state,
        )
